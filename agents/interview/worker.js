import { Room, LocalAudioTrack, RemoteAudioTrack, RoomEvent, TrackSource, TrackKind, AudioStream } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import dotenv from "dotenv";

import { DeepgramSTT } from "./stt.js";
import { AudioPublisher } from "../shared/audioPublisher.js";
import { generatePCM, generatePCMPipelined } from "./tts.js";
import { SessionBridge } from "./sessionBridge.js";
import { parseTTSResponse, checkTTSCache } from "../../lib/interview/interviewAgent.js";
import { mp3ToPCM } from "./mp3ToPCM.js";
import { agentLog } from "../../lib/logger.js";

// Spoken text for cached TTS phrase keys.
// The LLM emits tags like [great_answer] which parseTTSResponse strips out.
// In the LiveKit agent path we can't play MP3 files directly (need PCM),
// so we map each key to its spoken equivalent and TTS it inline.
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

// Improvement #5: Barge-in threshold — average absolute amplitude above which
// we treat incoming audio as intentional speech, not background noise.
// Int16 range is ±32767. 400 ≈ 1.2% of max — quiet speech is ~1000+.
const BARGE_IN_AMPLITUDE_THRESHOLD = 400;

// Improvement #5: Minimum word count for a barge-in to be processed.
// Prevents single-word noise ("Hmm") from interrupting long AI responses.
const BARGE_IN_MIN_WORDS = 3;

dotenv.config();

/**
 * Returns true if the Int16Array contains audio that is louder than
 * background noise — used to gate barge-in audio forwarding to STT.
 */
function hasSignificantAudio(int16Array) {
    if (!int16Array || int16Array.length === 0) return false;
    let sum = 0;
    // Sample every 4th value for performance on large buffers
    for (let i = 0; i < int16Array.length; i += 4) {
        sum += Math.abs(int16Array[i]);
    }
    return (sum / Math.ceil(int16Array.length / 4)) > BARGE_IN_AMPLITUDE_THRESHOLD;
}

export class InterviewAgentWorker {
    /**
     * @param {string} sessionId    - Unique room name / interview session
     * @param {Object} sessionCache - Ref to the server.js global sessionCache
     * @param {Object} agentWorkflow - Ref to the compiled LangGraph workflow
     * @param {Object} io           - Socket.io instance for UI updates (optional)
     * @param {string} candidateName - Candidate's name for the personalised intro
     * @param {number} maxQuestions  - Total questions in this interview (used in intro text)
     */
    constructor(sessionId, sessionCache, agentWorkflow, io, candidateName = "there", maxQuestions = 5) {
        this.sessionId = sessionId;
        this.sessionCache = sessionCache;
        this.sessionBridge = new SessionBridge(sessionId, sessionCache, agentWorkflow);
        this.io = io;
        this.candidateName = candidateName;
        this.maxQuestions = maxQuestions;

        this.room = new Room();
        this.stt = new DeepgramSTT();
        // Polly PCM is most reliable at 16kHz; publish the same rate to LiveKit source.
        this.audioPublisher = new AudioPublisher(16000, 1);

        this.isActive = false;
        this.processingTurn = false;

        // Tracks the last question spoken so we can repeat it on pardon/repeat requests
        this.currentQuestion = "";

        // Improvement #4: track when AI last finished speaking so we can compute
        // timeToAnswer = elapsed since AI finished → user's STT transcript received
        this.aiStoppedSpeakingAt = 0;
    }

    async start() {
        const startTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, 'AgentWorker starting');

        // 1. Start Deepgram STT (non-blocking — WebSocket handshake runs in background)
        const sttModel = process.env.DEEPGRAM_STT_MODEL || 'nova-3';
        agentLog.info({ sessionId: this.sessionId, model: sttModel }, '[1/4] Starting Deepgram STT...');
        this.stt.start();

        // 2. Generate LiveKit token
        const tokenTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, '[2/4] Generating LiveKit token...');
        const token = await this.generateToken();
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - tokenTs) }, '[2/4] LiveKit token ready');
        this.setupRoomEvents();

        // 3. Connect to LiveKit Room
        const connectTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, '[3/4] Connecting to LiveKit room...');
        await this.room.connect(process.env.LIVEKIT_URL, token);
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - connectTs) }, '[3/4] LiveKit room connected');

        // 4. Publish AI voice track
        const trackTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, '[4/4] Publishing AI audio track...');
        const track = LocalAudioTrack.createAudioTrack("ai-interviewer-audio", this.audioPublisher.source);
        await this.room.localParticipant.publishTrack(track, {
            name: "ai-interviewer-audio",
            source: TrackSource.SOURCE_MICROPHONE
        });
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - trackTs), totalMs: Math.round(performance.now() - startTs) }, '[4/4] AgentWorker ready');

        this.isActive = true;
    }

    async generateToken() {
        const at = new AccessToken(process.env.LIVEKIT_API_KEY, process.env.LIVEKIT_API_SECRET, {
            identity: "ai-interviewer",
            name: "AI Interviewer",
        });
        at.addGrant({
            roomJoin: true,
            room: this.sessionId,
            canPublish: true,
            canSubscribe: true,
        });
        return await at.toJwt();
    }

    setupRoomEvents() {
        this.room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
            agentLog.debug({ kind: track.kind, from: participant.identity }, 'Track subscribed');
            if (track.kind === TrackKind.KIND_AUDIO) {
                agentLog.info({ sessionId: this.sessionId, participant: participant.identity }, 'Subscribed to user audio');
                this.listenToUserAudio(track);
            }
        });

        this.room.on(RoomEvent.Disconnected, () => {
            agentLog.info({ sessionId: this.sessionId }, 'Disconnected from room');
            this.stop();
        });

        // STT now emits an object: { transcript, utteranceDurationMs, fillerWordCount }
        this.stt.on("transcript", async ({ transcript: text, utteranceDurationMs, fillerWordCount }) => {
            if (!this.isActive) return;

            // Improvement #5: Barge-in — user spoke while AI was still talking
            if (this.audioPublisher.isSpeaking) {
                const wordCount = text.trim().split(/\s+/).filter(w => w).length;
                if (wordCount >= BARGE_IN_MIN_WORDS) {
                    agentLog.info({
                        sessionId: this.sessionId,
                        transcript: text.substring(0, 80),
                        wordCount,
                    }, '🎤 Barge-in detected — interrupting AI');

                    // Stop AI audio immediately (AudioPublisher.stop() sets stopFlag=true
                    // which breaks the pushPCM loop on the very next 10ms chunk)
                    this.audioPublisher.stop();

                    // Brief pause to let the audio subsystem settle before we re-speak
                    await new Promise(r => setTimeout(r, 120));

                    const timeToAnswer = this.aiStoppedSpeakingAt > 0
                        ? Date.now() - this.aiStoppedSpeakingAt
                        : 0;

                    await this.handleUserTurn(text, {
                        utteranceDurationMs,
                        fillerWordCount,
                        timeToAnswer,
                        bargedIn: true,
                    });
                } else {
                    agentLog.debug({ sessionId: this.sessionId, wordCount }, 'Ignored short transcript during AI speech');
                }
                return;
            }

            if (this.processingTurn) return;

            const timeToAnswer = this.aiStoppedSpeakingAt > 0
                ? Date.now() - this.aiStoppedSpeakingAt
                : 0;

            await this.handleUserTurn(text, {
                utteranceDurationMs,
                fillerWordCount,
                timeToAnswer,
                bargedIn: false,
            });
        });
    }

    listenToUserAudio(track) {
        // Use AudioStream to receive PCM frames from the remote track.
        // AudioStream is an async iterable that yields AudioFrame objects.
        const audioStream = new AudioStream(track);

        (async () => {
            try {
                for await (const frame of audioStream) {
                    if (frame.data) {
                        if (!this.processingTurn && !this.audioPublisher.isSpeaking) {
                            // Normal listening — forward all audio to STT
                            this.stt.pushAudio(frame.data);
                        } else if (this.audioPublisher.isSpeaking) {
                            // Improvement #5: Barge-in — forward audio only when above noise floor.
                            // This lets Deepgram detect the user speaking while AI is talking
                            // without flooding it with silence/background noise frames.
                            if (hasSignificantAudio(frame.data)) {
                                this.stt.pushAudio(frame.data);
                            }
                        }
                    }
                }
                agentLog.debug({ sessionId: this.sessionId }, 'User audio stream ended');
            } catch (err) {
                agentLog.error({ sessionId: this.sessionId, err: err.message }, 'User audio stream error');
            }
        })();
    }

    /**
     * @param {string} transcript - The STT-transcribed user speech
     * @param {Object} acousticMeta - Behavioral context from STT/WebRTC layer
     */
    async handleUserTurn(transcript, acousticMeta = {}) {
        const turnStart = performance.now();
        this.processingTurn = true;

        // Safety: if LLM or TTS hangs, don't block the interview indefinitely.
        // After 30s we reset the flag so new user speech can be processed again.
        const safetyTimer = setTimeout(() => {
            if (this.processingTurn) {
                agentLog.warn({ sessionId: this.sessionId }, 'processingTurn safety timeout — resetting flag');
                this.processingTurn = false;
            }
        }, 30000);

        agentLog.info({
            sessionId: this.sessionId,
            transcript: transcript.substring(0, 150),
            words: transcript.trim().split(/\s+/).length,
            utteranceDurationMs: acousticMeta.utteranceDurationMs || 0,
            fillerWordCount: acousticMeta.fillerWordCount || 0,
            timeToAnswer: acousticMeta.timeToAnswer || 0,
            bargedIn: acousticMeta.bargedIn || false,
        }, '🎤 User said');

        // Notify UI that we heard something
        if (this.io) {
            this.io.to(this.sessionId).emit('ai_state', { state: 'thinking', text: 'Evaluating your answer...' });
            this.io.to(this.sessionId).emit('transcript_final', { role: 'user', text: transcript });
        }

        try {
            // ── Fast-path: pardon / repeat request ─────────────────────────────────
            // Intercept before LangGraph so we never accidentally advance state or
            // trigger the final report on the last question just because the user
            // asked us to repeat it.
            const lowerAns = transcript.trim().toLowerCase().replace(/[^a-z\s]/g, "").trim();
            if (this.currentQuestion && /\b(pardon|repeat|say again|say that again|can you repeat|what was the question|come again|once more)\b/.test(lowerAns)) {
                agentLog.info({ sessionId: this.sessionId, transcript: transcript.substring(0, 80) }, 'Pardon/repeat request — re-speaking current question');
                this.io?.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'Repeating question...' });
                await this._speakAndEmit(this.currentQuestion);
                this.io?.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Your turn...' });
                return;
            }

            // 1. Process via LangGraph (pass acoustic metadata for behavioral context)
            const result = await this.sessionBridge.processUserTranscript(transcript, acousticMeta);
            const processMs = Math.round(performance.now() - turnStart);
            agentLog.info({ sessionId: this.sessionId, processMs, intent: result.intent, done: result.done }, 'User turn processed');

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
                }, 'LangGraph turn result');
            }

            // 2. Notify UI of feedback/score
            if (this.io && result.evaluation) {
                this.io.to(this.sessionId).emit('ai_feedback', {
                    feedback: result.evaluation.feedback,
                    score: result.evaluation.score,
                    answerQuality: result.answerQuality
                });
            }

            // ── Improvement #1 & #2: New intent routing in worker ──────────────────

            // Intents fully handled inside handleEdgeCaseNode — speak feedback only
            const FEEDBACK_ONLY_INTENTS = ['confused', 'meta', 'irrelevant'];

            // Improvement #1: backchannel — candidate is thinking out loud, don't evaluate
            const THINKING_INTENTS = ['thinking_out_loud'];

            // Improvement #1: premature STT cutoff — wait silently for more speech
            const CUTOFF_INTENTS = ['premature_cutoff'];

            if (result.done) {
                // ── Interview complete ─────────────────────────────────────────────
                if (this.io) {
                    this.io.to(this.sessionId).emit('interview_done', {
                        report: result.finalReport,
                        topicScores: result.topicScores || {},
                        scores: result.scores || [],
                        questionsAsked: result.questionsAsked || 0
                    });
                }

                // Save interview results to DB via the callback set in routes/interview.js
                const session = this.sessionCache[this.sessionId];
                if (session?._onInterviewComplete) {
                    try {
                        await session._onInterviewComplete(result);
                    } catch (err) {
                        agentLog.error({ sessionId: this.sessionId, err: err.message }, 'Failed to save interview result');
                    }
                }

                // Play interview_outro.mp3 for normal completion, or
                // interview_stopped.mp3 when the user explicitly requested to stop.
                const isStoppedByUser = result.intent === 'stop' || result.intent === 'unwell';
                if (isStoppedByUser) {
                    await this._playFileOrFallback(
                        'interview_stopped',
                        "Of course. As you requested, we will end the interview here. Thank you so much for your time today. I wish you all the best."
                    );
                } else {
                    await this._playFileOrFallback(
                        'interview_outro',
                        "Thank you for completing the interview. That concludes all our questions. You can now review your full report on the screen. Well done and good luck!"
                    );
                }

                setTimeout(() => this.stop(), 2000);

            } else if (CUTOFF_INTENTS.includes(result.intent)) {
                // ── Premature STT cutoff — say nothing, just keep listening ────────
                // The candidate's sentence was cut off mid-thought. 700ms endpointing
                // greatly reduces this, but when it does happen we simply wait for more.
                agentLog.info({ sessionId: this.sessionId }, 'Premature cutoff — waiting for more speech');
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Please continue...' });
                }

            } else if (THINKING_INTENTS.includes(result.intent)) {
                // ── Thinking out loud — acknowledge briefly and wait ────────────────
                // Candidate is mid-thought. Speak a short backchannel phrase so they know
                // we're listening, then go back to listening without advancing question state.
                const rawFeedback = result.evaluation?.feedback || "[take_your_time]";
                const parsed = parseTTSResponse(rawFeedback);
                const phrases = (parsed.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
                const spokenText = (`${phrases} ${parsed.uniquePart}`).trim() || "Take your time.";

                agentLog.info({ sessionId: this.sessionId, spoken: spokenText }, 'Backchannel — thinking out loud');
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }
                await this._speakAndEmit(spokenText);
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Your turn...' });
                }

            } else if (FEEDBACK_ONLY_INTENTS.includes(result.intent)) {
                // ── confused / meta / irrelevant — speak response, stay on same question ──
                const feedbackText = result.evaluation?.feedback || "";
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }
                if (feedbackText) {
                    await this._speakAndEmit(feedbackText);
                }
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Your turn to speak...' });
                }

            } else {
                // ── Next question — speak feedback phrase + new question ────────────
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }

                // Parse TTS tags
                const parsedFeedback = parseTTSResponse(result.evaluation?.feedback || "");
                const parsedQuestion = parseTTSResponse(result.nextQuestion || "");

                const feedbackPhrases = (parsedFeedback.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
                const questionPhrases = (parsedQuestion.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");

                // Log phrase cache hits
                for (const k of (parsedFeedback.phraseKeys || [])) {
                    if (PHRASE_TEXT[k]) agentLog.info({ sessionId: this.sessionId, tag: k, phrase: PHRASE_TEXT[k] }, 'TTS phrase cache hit');
                }
                for (const k of (parsedQuestion.phraseKeys || [])) {
                    if (PHRASE_TEXT[k]) agentLog.info({ sessionId: this.sessionId, tag: k, phrase: PHRASE_TEXT[k] }, 'TTS phrase cache hit');
                }

                // Build two parts: short feedback phrase + longer question text
                const feedbackText = `${feedbackPhrases} ${parsedFeedback.uniquePart}`.trim();
                const questionText = `${questionPhrases} ${parsedQuestion.uniquePart}`.trim();

                // Track for pardon/repeat requests
                if (questionText) this.currentQuestion = questionText;

                // Start the AI turn — feedback + question are one logical turn
                this.io?.to(this.sessionId).emit('ai_speech', { action: 'start' });

                if (feedbackText && questionText) {
                    agentLog.info({ sessionId: this.sessionId, feedback: feedbackText.substring(0, 80), question: questionText.substring(0, 80) }, 'TTS parallel (feedback + question)');
                    // Start question TTS generation in background while we play feedback
                    const questionTTSPromise = generatePCM(questionText, "16000");

                    // Emit + play short feedback immediately
                    this.io?.to(this.sessionId).emit('ai_speech', { action: 'sentence', text: feedbackText });
                    const feedbackPcm = await generatePCM(feedbackText, "16000");
                    if (feedbackPcm) await this._playAudio(feedbackPcm);

                    // Emit + play question (PCM should be ready by now)
                    this.io?.to(this.sessionId).emit('ai_speech', { action: 'sentence', text: questionText });
                    const questionPcm = await questionTTSPromise;
                    if (questionPcm) await this._playAudio(questionPcm);
                } else {
                    const spokenText = `${feedbackText} ${questionText}`.trim();
                    if (spokenText) {
                        agentLog.info({ sessionId: this.sessionId, text: spokenText.substring(0, 80) }, 'TTS pipelined generation');
                        // start/end already handled — only emit sentences
                        await this._speakAndEmit(spokenText, false, false);
                    }
                }

                this.io?.to(this.sessionId).emit('ai_speech', { action: 'end' });

                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Listening...' });
                }
            }

            const totalMs = Math.round(performance.now() - turnStart);
            agentLog.info({ sessionId: this.sessionId, totalMs }, 'Turn complete');

        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'Turn error');
        } finally {
            clearTimeout(safetyTimer);
            this.processingTurn = false;
        }
    }

    /**
     * Sends PCM audio via LiveKit and records when AI finished speaking.
     */
    async _playAudio(pcm) {
        if (!pcm) return;
        await this.audioPublisher.pushPCM(pcm);
        this.aiStoppedSpeakingAt = Date.now();
    }

    /**
     * Speaks `text` sentence-by-sentence via pipelined TTS, emitting
     * `ai_speech` socket events so the frontend transcript updates in sync
     * with the actual audio playback.
     *
     * Events emitted:
     *   { action: 'start' }             — new AI turn beginning
     *   { action: 'sentence', text }    — one sentence about to play
     *   { action: 'end' }               — turn finished
     *
     * @param {string}  text        - Clean text (no [tags])
     * @param {boolean} emitStart   - Emit 'start' event (default true). Pass false
     *                                when chaining multiple calls inside one turn.
     * @param {boolean} emitEnd     - Emit 'end' event (default true).
     */
    async _speakAndEmit(text, emitStart = true, emitEnd = true) {
        if (!text) return;
        if (emitStart) this.io?.to(this.sessionId).emit('ai_speech', { action: 'start' });
        for await (const { pcm, text: sentence } of generatePCMPipelined(text, "16000")) {
            this.io?.to(this.sessionId).emit('ai_speech', { action: 'sentence', text: sentence });
            await this._playAudio(pcm);
        }
        if (emitEnd) this.io?.to(this.sessionId).emit('ai_speech', { action: 'end' });
    }

    /**
     * Plays interview_intro.mp3 with the candidate's name woven in via Polly,
     * then speaks the first question.
     * Called fire-and-forget from routes/interview.js instead of speak().
     */
    async speakIntro(firstQuestion) {
        if (!firstQuestion) return;
        this.processingTurn = true;
        const safetyTimer = setTimeout(() => {
            if (this.processingTurn) {
                agentLog.warn({ sessionId: this.sessionId }, 'speakIntro safety timeout — resetting');
                this.processingTurn = false;
            }
        }, 30000);
        try {
            // Personalised intro generated via Polly so the candidate's name is spoken correctly.
            // interview_intro.mp3 is a static file (no name), so we regenerate with the name here.
            const introText = `Hello, ${this.candidateName}! Welcome to your AI voice interview. I will ask you ${this.maxQuestions} questions based on the document you uploaded. Please answer each question clearly after I finish speaking. Let's begin!`;
            agentLog.info({ sessionId: this.sessionId, candidateName: this.candidateName }, 'Speaking personalised intro');
            // Intro + first question are one AI turn — emit start once, sentences as they play
            this.io?.to(this.sessionId).emit('ai_speech', { action: 'start' });
            for await (const { pcm, text: sentence } of generatePCMPipelined(introText, "16000")) {
                this.io?.to(this.sessionId).emit('ai_speech', { action: 'sentence', text: sentence });
                await this._playAudio(pcm);
            }
            // Speak the first question
            const parsed = parseTTSResponse(firstQuestion);
            const phrases = (parsed.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
            const spokenQuestion = (`${phrases} ${parsed.uniquePart}`).trim();
            if (spokenQuestion) this.currentQuestion = spokenQuestion;
            if (spokenQuestion) {
                for await (const { pcm, text: sentence } of generatePCMPipelined(spokenQuestion, "16000")) {
                    this.io?.to(this.sessionId).emit('ai_speech', { action: 'sentence', text: sentence });
                    await this._playAudio(pcm);
                }
            }
            this.io?.to(this.sessionId).emit('ai_speech', { action: 'end' });
        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'speakIntro error');
        } finally {
            clearTimeout(safetyTimer);
            this.processingTurn = false;
        }
    }

    /**
     * Tries to play a cached MP3 phrase through LiveKit (decoded via ffmpeg).
     * Falls back to generating the fallbackText via Polly TTS if ffmpeg is unavailable.
     *
     * @param {string} phraseKey    - Key matching an entry in TTS_PHRASE_CACHE (e.g. 'interview_outro')
     * @param {string} fallbackText - Text to synthesise via Polly if the MP3 can't be decoded
     */
    async _playFileOrFallback(phraseKey, fallbackText) {
        this.io?.to(this.sessionId).emit('ai_speech', { action: 'start' });
        const filePath = checkTTSCache(phraseKey);
        if (filePath) {
            const pcm = await mp3ToPCM(filePath, 16000);
            if (pcm) {
                agentLog.info({ sessionId: this.sessionId, phraseKey }, 'Playing cached MP3 phrase');
                this.io?.to(this.sessionId).emit('ai_speech', { action: 'sentence', text: fallbackText });
                await this._playAudio(pcm);
                this.io?.to(this.sessionId).emit('ai_speech', { action: 'end' });
                return;
            }
        }
        // ffmpeg not available or file missing — fall back to Polly
        agentLog.info({ sessionId: this.sessionId, phraseKey }, 'MP3 unavailable, using Polly fallback');
        await this._speakAndEmit(fallbackText, false, false);
        this.io?.to(this.sessionId).emit('ai_speech', { action: 'end' });
    }

    async speak(text) {
        if (!text) return;
        this.processingTurn = true;
        try {
            const parsed = parseTTSResponse(text);
            const phrases = (parsed.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
            const fullText = `${phrases} ${parsed.uniquePart}`.replace(/\s+/g, " ").trim();

            if (fullText) {
                agentLog.info({ sessionId: this.sessionId, text: fullText.substring(0, 80) }, 'Speaking (pipelined)');
                await this._speakAndEmit(fullText);
            }
        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'speak() error');
        } finally {
            this.processingTurn = false;
        }
    }

    stop() {
        agentLog.info({ sessionId: this.sessionId }, 'AgentWorker stopping');
        this.isActive = false;
        this.stt.stop();
        this.audioPublisher.stop();
        this.room.disconnect();
    }
}
