import { Room, LocalAudioTrack, RemoteAudioTrack, RoomEvent, TrackSource, TrackKind, AudioStream } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import dotenv from "dotenv";

import { DeepgramSTT } from "./stt.js";
import { AudioPublisher } from "./audioPublisher.js";
import { generatePCM, generatePCMPipelined } from "./tts.js";
import { SessionBridge } from "./sessionBridge.js";
import { parseTTSResponse } from "../lib/interview/interviewAgent.js";
import { agentLog } from "../lib/logger.js";

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

dotenv.config();

/**
 * Wraps raw 16-bit PCM in a minimal WAV header so the browser can play it
 * via a simple <audio> element. Returns a base64-encoded WAV string.
 */
function pcmToWavBase64(pcmInt16, sampleRate = 16000, channels = 1) {
    const bytesPerSample = 2;
    const dataSize = pcmInt16.length * bytesPerSample;
    const buffer = Buffer.alloc(44 + dataSize);

    // RIFF header
    buffer.write("RIFF", 0);
    buffer.writeUInt32LE(36 + dataSize, 4);
    buffer.write("WAVE", 8);

    // fmt chunk
    buffer.write("fmt ", 12);
    buffer.writeUInt32LE(16, 16);               // chunk size
    buffer.writeUInt16LE(1, 20);                // PCM format
    buffer.writeUInt16LE(channels, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(sampleRate * channels * bytesPerSample, 28); // byte rate
    buffer.writeUInt16LE(channels * bytesPerSample, 32);              // block align
    buffer.writeUInt16LE(16, 34);               // bits per sample

    // data chunk
    buffer.write("data", 36);
    buffer.writeUInt32LE(dataSize, 40);

    // Copy PCM samples
    const pcmBuf = Buffer.from(pcmInt16.buffer, pcmInt16.byteOffset, dataSize);
    pcmBuf.copy(buffer, 44);

    return buffer.toString("base64");
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
    }

    async start() {
        agentLog.info({ sessionId: this.sessionId }, 'AgentWorker starting');

        // 1. Start STT connection early (non-blocking WebSocket handshake)
        this.stt.start();

        // 2. Generate token + setup room events in parallel
        const token = await this.generateToken();
        this.setupRoomEvents();

        // 3. Connect to LiveKit Room
        await this.room.connect(process.env.LIVEKIT_URL, token);
        agentLog.info({ sessionId: this.sessionId }, 'AgentWorker connected to room');

        // 4. Publish AI voice track
        const track = LocalAudioTrack.createAudioTrack("ai-interviewer-audio", this.audioPublisher.source);
        await this.room.localParticipant.publishTrack(track, {
            name: "ai-interviewer-audio",
            source: TrackSource.SOURCE_MICROPHONE // properly imported enum
        });
        agentLog.info({ sessionId: this.sessionId }, 'AI audio track published');

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

        // Handle full transcript from STT
        this.stt.on("transcript", async (text) => {
            if (!this.isActive || this.processingTurn) return;

            // Basic VAD: don't process if the AI is currently speaking
            if (this.audioPublisher.isSpeaking) {
                agentLog.debug({ sessionId: this.sessionId }, 'Ignored transcript (AI is speaking)');
                return;
            }

            await this.handleUserTurn(text);
        });
    }

    listenToUserAudio(track) {
        // Use AudioStream to receive PCM frames from the remote track.
        // AudioStream is an async iterable that yields AudioFrame objects.
        const audioStream = new AudioStream(track);

        (async () => {
            try {
                for await (const frame of audioStream) {
                    // Only forward audio to STT when the agent is idle
                    if (!this.processingTurn && !this.audioPublisher.isSpeaking) {
                        if (frame.data) {
                            this.stt.pushAudio(frame.data);
                        }
                    }
                }
                agentLog.debug({ sessionId: this.sessionId }, 'User audio stream ended');
            } catch (err) {
                agentLog.error({ sessionId: this.sessionId, err: err.message }, 'User audio stream error');
            }
        })();
    }

    async handleUserTurn(transcript) {
        const turnStart = performance.now();
        this.processingTurn = true;

        // Notify UI that we heard something
        if (this.io) {
            this.io.to(this.sessionId).emit('ai_state', { state: 'thinking', text: 'Evaluating your answer...' });
            this.io.to(this.sessionId).emit('transcript_final', { role: 'user', text: transcript });
        }

        try {
            // 1. Process via LangGraph
            const result = await this.sessionBridge.processUserTranscript(transcript);
            const processMs = Math.round(performance.now() - turnStart);
            agentLog.info({ sessionId: this.sessionId, processMs, done: result.done }, 'User turn processed');

            // 2. Notify UI of feedback/score
            if (this.io && result.evaluation) {
                this.io.to(this.sessionId).emit('ai_feedback', {
                    feedback: result.evaluation.feedback,
                    score: result.evaluation.score,
                    answerQuality: result.answerQuality
                });
            }

            if (result.done) {
                // Interview over
                if (this.io) {
                    this.io.to(this.sessionId).emit('interview_done', { report: result.finalReport });
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
            } else {
                // Next question
                if (this.io) {
                    this.io.to(this.sessionId).emit('transcript_final', { role: 'ai', text: result.nextQuestion });
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }

                // Parse TTS tags
                const parsedFeedback = parseTTSResponse(result.evaluation.feedback || "");
                const parsedQuestion = parseTTSResponse(result.nextQuestion || "");

                const feedbackPhrases = (parsedFeedback.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
                const questionPhrases = (parsedQuestion.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");

                // Build two parts: short feedback phrase + longer question text
                const feedbackText = `${feedbackPhrases} ${parsedFeedback.uniquePart}`.trim();
                const questionText = `${questionPhrases} ${parsedQuestion.uniquePart}`.trim();

                // Play feedback phrase first (short, fast TTS) while question TTS generates
                if (feedbackText && questionText) {
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
     * Can be called manually to make the agent speak (e.g. for the very first question)
     */
    /**
     * Sends PCM audio via both LiveKit (WebRTC) and Socket.io (WAV fallback).
     * The client plays whichever path is available.
     */
    async _playAudio(pcm) {
        if (!pcm) return;
        // LiveKit WebRTC — primary audio path
        await this.audioPublisher.pushPCM(pcm);
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
