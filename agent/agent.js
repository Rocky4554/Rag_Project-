import { Room, LocalAudioTrack, RemoteAudioTrack, RoomEvent, TrackSource, TrackKind, AudioStream } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import dotenv from "dotenv";

import { DeepgramSTT } from "./stt.js";
import { AudioPublisher } from "./audioPublisher.js";
import { generatePCM } from "./tts.js";
import { SessionBridge } from "./sessionBridge.js";
import { parseTTSResponse } from "../lib/interviewAgent.js";

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
        console.log(`[AgentWorker] Starting for session: ${this.sessionId}`);

        // 1. Generate token for the AI agent (toJwt() is async in newer livekit-server-sdk)
        const token = await this.generateToken();

        // 2. Connect STT
        this.stt.start();

        // 3. Connect to LiveKit Room
        this.setupRoomEvents();
        await this.room.connect(process.env.LIVEKIT_URL, token);
        console.log(`[AgentWorker] Connected to room: ${this.sessionId}`);

        // 4. Publish AI voice track
        const track = LocalAudioTrack.createAudioTrack("ai-interviewer-audio", this.audioPublisher.source);
        await this.room.localParticipant.publishTrack(track, {
            name: "ai-interviewer-audio",
            source: TrackSource.SOURCE_MICROPHONE // properly imported enum
        });
        console.log(`[AgentWorker] Published AI audio track`);

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
            console.log(`[AgentWorker] TrackSubscribed: kind=${track.kind}, from=${participant.identity}`);
            if (track.kind === TrackKind.KIND_AUDIO) {
                console.log(`[AgentWorker] Subscribed to audio from: ${participant.identity}`);
                this.listenToUserAudio(track);
            }
        });

        this.room.on(RoomEvent.Disconnected, () => {
            console.log(`[AgentWorker] Disconnected from room.`);
            this.stop();
        });

        // Handle full transcript from STT
        this.stt.on("transcript", async (text) => {
            if (!this.isActive || this.processingTurn) return;
            
            // Basic VAD: don't process if the AI is currently speaking
            if (this.audioPublisher.isSpeaking) {
                console.log(`[AgentWorker] Ignored transcript (AI is speaking)`);
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
                console.log("[AgentWorker] User audio stream ended.");
            } catch (err) {
                console.error("[AgentWorker] User audio stream error:", err.message);
            }
        })();
    }

    async handleUserTurn(transcript) {
        this.processingTurn = true;
        
        // Notify UI that we heard something
        if (this.io) {
            this.io.to(this.sessionId).emit('ai_state', { state: 'thinking', text: 'Evaluating your answer...' });
            this.io.to(this.sessionId).emit('transcript_final', { role: 'user', text: transcript });
        }

        try {
            // 1. Process via LangGraph
            const result = await this.sessionBridge.processUserTranscript(transcript);
            
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

                const pcm = await generatePCM("Thank you for your time. The interview is now complete. You can review your report on the screen.");
                if (pcm) await this._playAudio(pcm);

                setTimeout(() => this.stop(), 2000);
            } else {
                // Next question
                if (this.io) {
                    this.io.to(this.sessionId).emit('transcript_final', { role: 'ai', text: result.nextQuestion });
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }

                // Parse TTS tags and build full spoken text (phrase text + unique part)
                const parsedFeedback = parseTTSResponse(result.evaluation.feedback || "");
                const parsedQuestion = parseTTSResponse(result.nextQuestion || "");

                const feedbackPhrases = (parsedFeedback.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");
                const questionPhrases = (parsedQuestion.phraseKeys || []).map(k => PHRASE_TEXT[k] || "").join(" ");

                const spokenText = `${feedbackPhrases} ${parsedFeedback.uniquePart} ${questionPhrases} ${parsedQuestion.uniquePart}`.replace(/\s+/g, " ").trim();

                if (spokenText) {
                    const pcmData = await generatePCM(spokenText, "16000");
                    if (pcmData) {
                        await this._playAudio(pcmData);
                    }
                }

                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Listening...' });
                }
            }

        } catch (err) {
            console.error(`[AgentWorker] Turn error:`, err);
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
                console.log(`[AgentWorker] Speaking: "${fullText.substring(0, 80)}..."`);
                const pcm = await generatePCM(fullText, "16000");
                if (pcm) {
                    await this._playAudio(pcm);
                } else {
                    console.error(`[AgentWorker] TTS returned null — Polly may have failed. Check AWS credentials/region.`);
                }
            }
        } catch (err) {
            console.error(`[AgentWorker] speak() error:`, err);
        } finally {
            this.processingTurn = false;
        }
    }

    stop() {
        console.log(`[AgentWorker] Stopping session: ${this.sessionId}`);
        this.isActive = false;
        this.stt.stop();
        this.audioPublisher.stop();
        this.room.disconnect();
    }
}