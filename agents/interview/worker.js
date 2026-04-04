import dotenv from "dotenv";

import { VoicePipelineWorker } from "../shared/voicePipelineWorker.js";
import { estimateWordTimings } from "./tts.js";
import { SessionBridge } from "./sessionBridge.js";
import { parseTTSResponse, checkTTSCache } from "../../lib/interview/interviewAgent.js";
import { mp3ToPCM } from "./mp3ToPCM.js";
import { agentLog } from "../../lib/logger.js";

dotenv.config();

// Spoken text for cached TTS phrase keys.
const PHRASE_TEXT = {
    "lets_move_on":      "Let's move on.",
    "great_answer":      "Great answer!",
    "good_effort":       "Good effort.",
    "take_your_time":    "Take your time.",
    "no_worries":        "No worries.",
    "interesting":       "Interesting.",
    "next_question":     "Next question.",
    "final_question":    "This is the final question.",
    "thats_okay":        "That's okay.",
    "thanks_for_time":   "Thanks for your time.",
    "interview_intro":   "Welcome to your interview.",
    "interview_outro":   "That concludes our interview.",
    "interview_stopped": "The interview has been stopped.",
    "out_of_context":    "Let's stay on topic.",
};

const FEEDBACK_ONLY_INTENTS = ['confused', 'meta', 'irrelevant'];
const THINKING_INTENTS = ['thinking_out_loud'];
const CUTOFF_INTENTS = ['premature_cutoff'];

export class InterviewAgentWorker extends VoicePipelineWorker {
    /**
     * @param {string} sessionId     - Unique room name / interview session
     * @param {Object} sessionCache  - Ref to the server.js global sessionCache
     * @param {Object} agentWorkflow - Ref to the compiled LangGraph workflow
     * @param {Object} io            - Socket.io instance for UI updates (optional)
     * @param {string} candidateName - Candidate's name for the personalised intro
     * @param {number} maxQuestions  - Total questions in this interview
     */
    constructor(sessionId, sessionCache, agentWorkflow, io, candidateName = "there", maxQuestions = 5) {
        super({
            sessionId,
            io,
            identity: "ai-interviewer",
            displayName: "AI Interviewer",
            sampleRate: 16000,
        });

        this.sessionCache = sessionCache;
        this.sessionBridge = new SessionBridge(sessionId, sessionCache, agentWorkflow);
        this.candidateName = candidateName;
        this.maxQuestions = maxQuestions;
        this.currentQuestion = "";
    }

    // ── Core: handle each user turn ─────────────────────────────

    async onUserTranscript(transcript, acousticMeta) {
        // ── Fast-path: pardon / repeat request ──────────────────
        const lowerAns = transcript.trim().toLowerCase().replace(/[^a-z\s]/g, "").trim();
        if (this.currentQuestion && /\b(pardon|repeat|say again|say that again|can you repeat|what was the question|come again|once more)\b/.test(lowerAns)) {
            agentLog.info({ sessionId: this.sessionId, transcript: transcript.substring(0, 80) }, 'Pardon/repeat request');
            return {
                speakCustom: async () => {
                    await this._speakAndEmit(this.currentQuestion);
                },
            };
        }

        // ── Process via LangGraph ───────────────────────────────
        const result = await this.sessionBridge.processUserTranscript(transcript, acousticMeta);
        agentLog.info({ sessionId: this.sessionId, intent: result.intent, done: result.done }, 'LangGraph result');

        if (!result.done && result.evaluation) {
            agentLog.info({
                sessionId: this.sessionId,
                intent: result.intent,
                quality: result.answerQuality,
                score: result.evaluation?.score,
                topic: result.topicTag,
                difficulty: result.difficultyLevel,
                nextDifficulty: result.evaluation?.nextDifficulty,
                questionNum: result.questionsAsked,
            }, 'Turn details');
        }

        // Build emitEvents for feedback/score
        const emitEvents = [];
        if (result.evaluation) {
            emitEvents.push({
                event: 'ai_feedback',
                data: {
                    feedback: result.evaluation.feedback,
                    score: result.evaluation.score,
                    answerQuality: result.answerQuality,
                },
            });
        }

        // ── Interview complete ──────────────────────────────────
        if (result.done) {
            emitEvents.push({
                event: 'interview_done',
                data: {
                    report: result.finalReport,
                    topicScores: result.topicScores || {},
                    scores: result.scores || [],
                    questionsAsked: result.questionsAsked || 0,
                },
            });

            return {
                done: true,
                emitEvents,
                speakCustom: async () => {
                    // Save interview results to DB
                    const session = this.sessionCache[this.sessionId];
                    if (session?._onInterviewComplete) {
                        try { await session._onInterviewComplete(result); }
                        catch (err) { agentLog.error({ sessionId: this.sessionId, err: err.message }, 'Failed to save interview result'); }
                    }

                    const isStoppedByUser = result.intent === 'stop' || result.intent === 'unwell';
                    if (isStoppedByUser) {
                        await this._playFileOrFallback('interview_stopped',
                            "Of course. As you requested, we will end the interview here. Thank you so much for your time today. I wish you all the best.");
                    } else {
                        await this._playFileOrFallback('interview_outro',
                            "Thank you for completing the interview. That concludes all our questions. You can now review your full report on the screen. Well done and good luck!");
                    }
                },
            };
        }

        // ── Premature cutoff — stay silent ──────────────────────
        if (CUTOFF_INTENTS.includes(result.intent)) {
            agentLog.info({ sessionId: this.sessionId }, 'Premature cutoff — waiting');
            return { silent: true };
        }

        // ── Thinking out loud — brief backchannel ───────────────
        if (THINKING_INTENTS.includes(result.intent)) {
            const rawFeedback = result.evaluation?.feedback || "[take_your_time]";
            const parsed = parseTTSResponse(rawFeedback);
            const phrases = (parsed.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
            const spokenText = (`${phrases} ${parsed.uniquePart}`).trim() || "Take your time.";
            agentLog.info({ sessionId: this.sessionId, spoken: spokenText }, 'Backchannel');
            return { segments: [{ text: spokenText }], emitEvents };
        }

        // ── Confused / meta / irrelevant — speak feedback only ──
        if (FEEDBACK_ONLY_INTENTS.includes(result.intent)) {
            const feedbackText = result.evaluation?.feedback || "";
            if (feedbackText) {
                return { segments: [{ text: feedbackText }], emitEvents };
            }
            return { silent: true, emitEvents };
        }

        // ── Normal answer — feedback + next question ────────────
        const parsedFeedback = parseTTSResponse(result.evaluation?.feedback || "");
        const parsedQuestion = parseTTSResponse(result.nextQuestion || "");

        const feedbackPhrases = (parsedFeedback.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
        const questionPhrases = (parsedQuestion.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");

        for (const k of (parsedFeedback.phraseKeys || [])) {
            if (PHRASE_TEXT[k]) agentLog.info({ sessionId: this.sessionId, tag: k }, 'TTS phrase cache hit');
        }

        const feedbackText = `${feedbackPhrases} ${parsedFeedback.uniquePart}`.trim();
        const questionText = `${questionPhrases} ${parsedQuestion.uniquePart}`.trim();

        if (questionText) this.currentQuestion = questionText;

        return {
            emitEvents,
            speakCustom: async () => {
                this._emitToRoom('ai_speech', { action: 'start' });

                if (feedbackText && questionText) {
                    agentLog.info({ sessionId: this.sessionId, feedback: feedbackText.substring(0, 80), question: questionText.substring(0, 80) }, 'TTS parallel');
                    // Speak feedback with subtitles, then question with subtitles
                    await this._speakAndEmit(feedbackText, false, false);
                    await this._speakAndEmit(questionText, false, false);
                } else {
                    const spokenText = `${feedbackText} ${questionText}`.trim();
                    if (spokenText) {
                        await this._speakAndEmit(spokenText, false, false);
                    }
                }

                this._emitToRoom('ai_speech', { action: 'end' });
            },
        };
    }

    // ── Interview-specific methods ──────────────────────────────

    /**
     * Speak personalised intro + first question.
     * Called fire-and-forget from routes/interview.js.
     */
    async speakIntro(firstQuestion) {
        if (!firstQuestion) return;
        this.processingTurn = true;
        const safetyTimer = setTimeout(() => {
            if (this.processingTurn) {
                agentLog.warn({ sessionId: this.sessionId }, 'speakIntro safety timeout');
                this.processingTurn = false;
            }
        }, 30000);
        try {
            const introText = `Hello, ${this.candidateName}! Welcome to your AI voice interview. I will ask you ${this.maxQuestions} questions based on the document you uploaded. Please answer each question clearly after I finish speaking. Let's begin!`;
            agentLog.info({ sessionId: this.sessionId, candidateName: this.candidateName }, 'Speaking intro');

            // Speak intro with subtitles
            await this._speakAndEmit(introText, true, false);

            // Speak first question with subtitles
            const parsed = parseTTSResponse(firstQuestion);
            const phrases = (parsed.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
            const spokenQuestion = (`${phrases} ${parsed.uniquePart}`).trim();
            if (spokenQuestion) this.currentQuestion = spokenQuestion;
            if (spokenQuestion) {
                await this._speakAndEmit(spokenQuestion, false, false);
            }
            this._emitToRoom('ai_speech', { action: 'end' });
        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'speakIntro error');
        } finally {
            clearTimeout(safetyTimer);
            this.processingTurn = false;
        }
    }

    /**
     * Tries to play a cached MP3 phrase, falls back to TTS.
     */
    async _playFileOrFallback(phraseKey, fallbackText) {
        this._emitToRoom('ai_speech', { action: 'start' });
        const filePath = checkTTSCache(phraseKey);
        if (filePath) {
            const pcm = await mp3ToPCM(filePath, 16000);
            if (pcm) {
                agentLog.info({ sessionId: this.sessionId, phraseKey }, 'Playing cached MP3');
                this._emitToRoom('ai_speech', { action: 'sentence', text: fallbackText });
                // Emit subtitle for cached audio too
                const marks = estimateWordTimings(fallbackText);
                if (marks.length > 0) this._emitToRoom('ai_subtitle', { words: marks, text: fallbackText });
                await this._playAudio(pcm);
                this._emitToRoom('ai_speech', { action: 'end' });
                return;
            }
        }
        agentLog.info({ sessionId: this.sessionId, phraseKey }, 'MP3 unavailable, using TTS fallback');
        await this._speakAndEmit(fallbackText, false, false);
        this._emitToRoom('ai_speech', { action: 'end' });
    }

    /** Speak arbitrary text with TTS tag parsing. */
    async speak(text) {
        if (!text) return;
        this.processingTurn = true;
        try {
            const parsed = parseTTSResponse(text);
            const phrases = (parsed.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
            const fullText = `${phrases} ${parsed.uniquePart}`.replace(/\s+/g, " ").trim();
            if (fullText) {
                agentLog.info({ sessionId: this.sessionId, text: fullText.substring(0, 80) }, 'Speaking');
                await this._speakAndEmit(fullText);
            }
        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'speak() error');
        } finally {
            this.processingTurn = false;
        }
    }
}
