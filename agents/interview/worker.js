import { Room, LocalAudioTrack, RemoteAudioTrack, RoomEvent, TrackSource, TrackKind, AudioStream } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import dotenv from "dotenv";

import { DeepgramSTT } from "./stt.js";
import { AudioPublisher } from "../shared/audioPublisher.js";
import { generatePCM, generatePCMPipelined } from "./tts.js";
import { SessionBridge } from "./sessionBridge.js";
import { parseTTSResponse } from "../../lib/interview/interviewAgent.js";
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
     * @param {string} sessionId - Unique room name / interview session
     * @param {Object} sessionCache - Ref to the server.js global sessionCache
     * @param {Object} agentWorkflow - Ref to the compiled LangGraph workflow
     * @param {Object} io - Socket.io instance for UI updates (optional)
     */
    constructor(sessionId, sessionCache, agentWorkflow, io) {
        this.sessionId = sessionId;
        this.sessionCache = sessionCache;
        this.sessionBridge = new SessionBridge(sessionId, sessionCache, agentWorkflow);
        this.io = io;

        this.room = new Room();
        this.stt = new DeepgramSTT();
        // Polly PCM is most reliable at 16kHz; publish the same rate to LiveKit source.
        this.audioPublisher = new AudioPublisher(16000, 1);

        this.isActive = false;
        this.processingTurn = false;

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

                for await (const { pcm } of generatePCMPipelined("Thank you for your time. The interview is now complete. You can review your report on the screen.", "16000")) {
                    await this._playAudio(pcm);
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
                for await (const { pcm } of generatePCMPipelined(spokenText, "16000")) {
                    await this._playAudio(pcm);
                }
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Your turn...' });
                }

            } else if (FEEDBACK_ONLY_INTENTS.includes(result.intent)) {
                // ── confused / meta / irrelevant — speak response, stay on same question ──
                const feedbackText = result.evaluation?.feedback || "";
                if (this.io) {
                    this.io.to(this.sessionId).emit('transcript_final', { role: 'ai', text: feedbackText });
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }
                if (feedbackText) {
                    for await (const { pcm } of generatePCMPipelined(feedbackText, "16000")) {
                        await this._playAudio(pcm);
                    }
                }
                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Your turn to speak...' });
                }

            } else {
                // ── Next question — speak feedback phrase + new question ────────────
                if (this.io) {
                    this.io.to(this.sessionId).emit('transcript_final', { role: 'ai', text: result.nextQuestion });
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

                // Play feedback phrase first (short, fast TTS) while question TTS generates
                if (feedbackText && questionText) {
                    agentLog.info({ sessionId: this.sessionId, feedback: feedbackText.substring(0, 80), question: questionText.substring(0, 80) }, 'TTS parallel (feedback + question)');
                    // Start question TTS generation in background while we play feedback
                    const questionTTSPromise = generatePCM(questionText, "16000");

                    // Play short feedback immediately
                    const feedbackPcm = await generatePCM(feedbackText, "16000");
                    if (feedbackPcm) await this._playAudio(feedbackPcm);

                    // Question PCM should be ready by now (or soon)
                    const questionPcm = await questionTTSPromise;
                    if (questionPcm) await this._playAudio(questionPcm);
                } else {
                    const spokenText = `${feedbackText} ${questionText}`.trim();
                    if (spokenText) {
                        agentLog.info({ sessionId: this.sessionId, text: spokenText.substring(0, 80) }, 'TTS pipelined generation');
                        for await (const { pcm } of generatePCMPipelined(spokenText, "16000")) {
                            await this._playAudio(pcm);
                        }
                    }
                }

                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Listening...' });
                }
            }

            const totalMs = Math.round(performance.now() - turnStart);
            agentLog.info({ sessionId: this.sessionId, totalMs }, 'Turn complete');

        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'Turn error');
        } finally {
            this.processingTurn = false;
        }
    }

    /**
     * Sends PCM audio via both LiveKit (WebRTC) and records when AI finished speaking.
     * Improvement #4: aiStoppedSpeakingAt timestamp is used to compute timeToAnswer.
     */
    async _playAudio(pcm) {
        if (!pcm) return;
        // LiveKit WebRTC — primary audio path
        await this.audioPublisher.pushPCM(pcm);
        // Record when AI last finished a chunk — used to compute timeToAnswer
        this.aiStoppedSpeakingAt = Date.now();
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
                // Stream sentence-by-sentence: play first sentence ASAP while rest generates
                for await (const { pcm } of generatePCMPipelined(fullText, "16000")) {
                    await this._playAudio(pcm);
                }
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
