/**
 * VoicePipelineWorker — Reusable base class for voice agents.
 *
 * Handles the full voice pipeline:
 *   User Mic → LiveKit → Deepgram STT → onUserTranscript() → TTS → LiveKit → Speaker
 *
 * Includes: barge-in detection, transcript streaming via socket.io,
 * Netflix-style subtitle sync, pipelined TTS (Polly/Deepgram/ElevenLabs).
 *
 * To build a new voice agent, extend this class and implement onUserTranscript().
 *
 * @example
 * class MyAgent extends VoicePipelineWorker {
 *   async onUserTranscript(transcript, acousticMeta) {
 *     const reply = await myBackend.respond(transcript);
 *     return { segments: [{ text: reply }] };
 *   }
 * }
 */
import { Room, LocalAudioTrack, RoomEvent, TrackSource, TrackKind, AudioStream } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import dotenv from "dotenv";

import { DeepgramSTT } from "../interview/stt.js";
import { AudioPublisher } from "./audioPublisher.js";
import { generatePCM, generatePCMPipelined, getSpeechMarks, estimateWordTimings } from "../interview/tts.js";
import { agentLog } from "../../lib/logger.js";

dotenv.config();

// ── Barge-in utilities ──────────────────────────────────────────

/**
 * Returns true if the Int16Array contains audio louder than background noise.
 * Used to gate barge-in audio forwarding to STT.
 */
function hasSignificantAudio(int16Array, threshold) {
    if (!int16Array || int16Array.length === 0) return false;
    let sum = 0;
    for (let i = 0; i < int16Array.length; i += 4) {
        sum += Math.abs(int16Array[i]);
    }
    return (sum / Math.ceil(int16Array.length / 4)) > threshold;
}

// ── Base class ──────────────────────────────────────────────────

export class VoicePipelineWorker {
    /**
     * @param {Object} opts
     * @param {string} opts.sessionId         - LiveKit room name / session ID
     * @param {Object} [opts.io]              - Socket.io instance for UI events
     * @param {string} [opts.identity]        - LiveKit participant identity
     * @param {string} [opts.displayName]     - LiveKit participant display name
     * @param {number} [opts.sampleRate]      - TTS output sample rate (default 16000)
     * @param {number} [opts.bargeInMinWords] - Min words to trigger barge-in (default 3)
     * @param {number} [opts.bargeInThreshold]- Amplitude threshold for barge-in (default 400)
     * @param {number} [opts.safetyTimeoutMs] - Turn processing timeout (default 30000)
     */
    constructor({
        sessionId,
        io = null,
        identity = "voice-agent",
        displayName = "Voice Agent",
        sampleRate = 16000,
        bargeInMinWords = 3,
        bargeInThreshold = 400,
        safetyTimeoutMs = 30000,
    }) {
        this.sessionId = sessionId;
        this.io = io;
        this.identity = identity;
        this.displayName = displayName;
        this.sampleRate = sampleRate;
        this.bargeInMinWords = bargeInMinWords;
        this.bargeInThreshold = bargeInThreshold;
        this.safetyTimeoutMs = safetyTimeoutMs;

        this.room = new Room();
        this.stt = new DeepgramSTT();
        this.audioPublisher = new AudioPublisher(sampleRate, 1);

        this.isActive = false;
        this.processingTurn = false;
        this.aiStoppedSpeakingAt = 0;
    }

    // ── Lifecycle ───────────────────────────────────────────────

    async start() {
        const startTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, 'VoicePipeline starting');

        // 1. Start Deepgram STT
        agentLog.info({ sessionId: this.sessionId }, '[1/4] Starting Deepgram STT...');
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
        agentLog.info({ sessionId: this.sessionId }, '[4/4] Publishing audio track...');
        const trackName = `${this.identity}-audio`;
        const track = LocalAudioTrack.createAudioTrack(trackName, this.audioPublisher.source);
        await this.room.localParticipant.publishTrack(track, {
            name: trackName,
            source: TrackSource.SOURCE_MICROPHONE,
        });
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - trackTs), totalMs: Math.round(performance.now() - startTs) }, '[4/4] VoicePipeline ready');

        this.isActive = true;
    }

    async generateToken() {
        const at = new AccessToken(process.env.LIVEKIT_API_KEY, process.env.LIVEKIT_API_SECRET, {
            identity: this.identity,
            name: this.displayName,
        });
        at.addGrant({
            roomJoin: true,
            room: this.sessionId,
            canPublish: true,
            canSubscribe: true,
        });
        return await at.toJwt();
    }

    stop() {
        agentLog.info({ sessionId: this.sessionId }, 'VoicePipeline stopping');
        this.isActive = false;
        this.stt.stop();
        this.audioPublisher.stop();
        this.room.disconnect();
    }

    // ── Room & STT event wiring ─────────────────────────────────

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

        // STT transcript handler with barge-in support
        this.stt.on("transcript", async ({ transcript: text, utteranceDurationMs, fillerWordCount }) => {
            if (!this.isActive) return;

            // Barge-in: user spoke while AI was still talking
            if (this.audioPublisher.isSpeaking) {
                const wordCount = text.trim().split(/\s+/).filter(w => w).length;
                if (wordCount >= this.bargeInMinWords) {
                    agentLog.info({ sessionId: this.sessionId, transcript: text.substring(0, 80), wordCount }, 'Barge-in detected');
                    this.audioPublisher.stop();
                    await new Promise(r => setTimeout(r, 120));

                    const timeToAnswer = this.aiStoppedSpeakingAt > 0 ? Date.now() - this.aiStoppedSpeakingAt : 0;
                    await this._handleTranscript(text, { utteranceDurationMs, fillerWordCount, timeToAnswer, bargedIn: true });
                } else {
                    agentLog.debug({ sessionId: this.sessionId, wordCount }, 'Ignored short transcript during AI speech');
                }
                return;
            }

            if (this.processingTurn) return;

            const timeToAnswer = this.aiStoppedSpeakingAt > 0 ? Date.now() - this.aiStoppedSpeakingAt : 0;
            await this._handleTranscript(text, { utteranceDurationMs, fillerWordCount, timeToAnswer, bargedIn: false });
        });
    }

    listenToUserAudio(track) {
        const audioStream = new AudioStream(track);
        (async () => {
            try {
                for await (const frame of audioStream) {
                    if (frame.data) {
                        if (!this.processingTurn && !this.audioPublisher.isSpeaking) {
                            this.stt.pushAudio(frame.data);
                        } else if (this.audioPublisher.isSpeaking) {
                            if (hasSignificantAudio(frame.data, this.bargeInThreshold)) {
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

    // ── Turn processing ─────────────────────────────────────────

    /**
     * Internal turn handler. Wraps onUserTranscript() with:
     * - processingTurn guard
     * - safety timeout
     * - socket.io event emission (thinking, transcript, speaking, listening)
     * - response processing (segments / speakCustom / done / silent)
     */
    async _handleTranscript(transcript, acousticMeta) {
        const turnStart = performance.now();
        this.processingTurn = true;

        const safetyTimer = setTimeout(() => {
            if (this.processingTurn) {
                agentLog.warn({ sessionId: this.sessionId }, 'Turn safety timeout — resetting');
                this.processingTurn = false;
            }
        }, this.safetyTimeoutMs);

        agentLog.info({
            sessionId: this.sessionId,
            transcript: transcript.substring(0, 150),
            words: transcript.trim().split(/\s+/).length,
            ...acousticMeta,
        }, 'User said');

        // Notify UI
        this._emitToRoom('ai_state', { state: 'thinking', text: 'Processing...' });
        this._emitToRoom('transcript_final', { role: 'user', text: transcript });

        try {
            // Call the subclass implementation
            const response = await this.onUserTranscript(transcript, acousticMeta);

            // Silent — don't speak, keep listening (e.g., premature cutoff)
            if (response.silent) {
                this._emitToRoom('ai_state', { state: 'listening', text: 'Listening...' });
                return;
            }

            // Emit any custom socket events (ai_feedback, interview_done, etc.)
            if (response.emitEvents) {
                for (const { event, data } of response.emitEvents) {
                    this._emitToRoom(event, data);
                }
            }

            // Speak the response
            if (response.speakCustom) {
                // Custom speech logic (e.g., parallel feedback + question TTS)
                this._emitToRoom('ai_state', { state: 'speaking', text: 'Speaking...' });
                await response.speakCustom();
            } else if (response.segments?.length > 0) {
                // Default: speak segments sequentially
                this._emitToRoom('ai_state', { state: 'speaking', text: 'Speaking...' });
                this._emitToRoom('ai_speech', { action: 'start' });
                for (const seg of response.segments) {
                    await this._speakAndEmit(seg.text, false, false);
                }
                this._emitToRoom('ai_speech', { action: 'end' });
            }

            // Post-speak callback
            if (response.afterSpeak) await response.afterSpeak();

            // Done — stop the session after a short delay
            if (response.done) {
                setTimeout(() => this.stop(), 2000);
            } else {
                this._emitToRoom('ai_state', { state: 'listening', text: 'Listening...' });
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

    // ── Extension point — MUST be overridden by subclass ────────

    /**
     * Process the user's speech and return what the agent should do.
     *
     * @param {string} transcript    - The STT-transcribed user speech
     * @param {Object} acousticMeta  - { utteranceDurationMs, fillerWordCount, timeToAnswer, bargedIn }
     *
     * @returns {Object} response:
     *   segments:    [{ text }]            - Things to speak (sequentially)
     *   speakCustom: async () => void      - OR custom speech logic (for parallel TTS, etc.)
     *   done:        boolean               - End session after speaking
     *   silent:      boolean               - Don't speak, keep listening
     *   emitEvents:  [{ event, data }]     - Custom socket.io events to emit
     *   afterSpeak:  async () => void      - Callback after speaking completes
     */
    async onUserTranscript(transcript, acousticMeta) {
        throw new Error('Subclass must implement onUserTranscript(transcript, acousticMeta)');
    }

    // ── Audio playback ──────────────────────────────────────────

    /** Push raw PCM to LiveKit and record when AI finished speaking. */
    async _playAudio(pcm) {
        if (!pcm) return;
        await this.audioPublisher.pushPCM(pcm);
        this.aiStoppedSpeakingAt = Date.now();
    }

    /**
     * Speak text via pipelined TTS with socket.io transcript + subtitle sync.
     * Handles all three TTS providers transparently.
     */
    async _speakAndEmit(text, emitStart = true, emitEnd = true) {
        if (!text) return;
        if (emitStart) this._emitToRoom('ai_speech', { action: 'start' });

        const provider = (process.env.INTERVIEW_TTS_PROVIDER || "polly").toLowerCase();

        const marksPromise = provider === "deepgram"
            ? Promise.resolve(estimateWordTimings(text))
            : getSpeechMarks(text).catch(() => estimateWordTimings(text));

        let textEmitted = false;
        let firstChunk = true;

        for await (const { pcm, text: chunkText } of generatePCMPipelined(text, String(this.sampleRate))) {
            if (chunkText && !textEmitted) {
                this._emitToRoom('ai_speech', { action: 'sentence', text: chunkText });
                textEmitted = true;
            } else if (chunkText) {
                this._emitToRoom('ai_speech', { action: 'sentence', text: chunkText });
            }

            if (firstChunk) {
                firstChunk = false;
                const marks = await marksPromise;
                if (marks.length > 0) {
                    this._emitToRoom('ai_subtitle', { words: marks, text });
                }
            }

            await this._playAudio(pcm);
        }

        if (emitEnd) this._emitToRoom('ai_speech', { action: 'end' });
    }

    // ── Socket.io helper ────────────────────────────────────────

    _emitToRoom(event, data) {
        if (this.io) this.io.to(this.sessionId).emit(event, data);
    }
}
