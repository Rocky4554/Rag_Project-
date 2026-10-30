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
import { AsyncLocalStorage } from "node:async_hooks";

// Carries the current turn through every await of its handler, so speech
// started by that turn can tell it was cancelled by a barge-in.
const turnContext = new AsyncLocalStorage();

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
        displayName = "Conversational AI",
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

        // Incremented on every barge-in. Each _speakAndEmit captures the epoch
        // at start and bails the moment it changes — so an interrupt cancels the
        // in-flight utterance, not just the single PCM chunk.
        this._speechEpoch = 0;

        // Turns run one at a time so two LangGraph invokes never race on the
        // same thread_id. The running turn is tracked so a barge-in can cancel
        // everything it still has left to say (e.g. the question after feedback).
        this._turnQueue = Promise.resolve();
        this._currentTurn = null;
    }

    /** Cancel all in-flight TTS playback: bumps the epoch so any running
     *  _speakAndEmit loop breaks, cancels the rest of the current turn's
     *  speech, and halts the current audio chunk. */
    _interruptSpeech() {
        this._speechEpoch++;
        if (this._currentTurn) this._currentTurn.cancelled = true;
        this.audioPublisher.stop();
    }

    /** True when the turn this code runs in was cancelled by a barge-in. */
    _turnCancelled() {
        return !!turnContext.getStore()?.cancelled;
    }

    /** Queue a user utterance; it runs after any turn still in progress. */
    _enqueueTurn(transcript, acousticMeta) {
        return this._enqueueTask(() => this._handleTranscript(transcript, acousticMeta));
    }

    /**
     * Run `fn` as a turn: after the previous turn, inside its own turn context
     * (so a barge-in can cancel its speech), guarded by the safety timeout.
     * @param {boolean} [opts.requireActive] - skip if the agent was stopped (false for the intro,
     *                                         which may be queued while LiveKit is still connecting)
     */
    _enqueueTask(fn, { requireActive = true } = {}) {
        const turn = { cancelled: false, timedOut: false };
        this._turnQueue = this._turnQueue.then(async () => {
            if (requireActive && !this.isActive) return;
            this._currentTurn = turn;
            let timer;
            const timeout = new Promise(resolve => {
                const arm = (ms) => {
                    timer = setTimeout(() => {
                        // Audio still playing means the turn is progressing, not hung
                        if (this.audioPublisher.isSpeaking) return arm(5000);
                        agentLog.warn({ sessionId: this.sessionId }, 'Turn safety timeout — abandoning turn');
                        // Silence whatever the stalled turn says later, and reopen the mic:
                        // audio isn't forwarded to STT while processingTurn is set.
                        turn.cancelled = true;
                        turn.timedOut = true;
                        this.processingTurn = false;
                        this._emitToRoom('ai_state', { state: 'listening', text: 'Listening...' });
                        resolve();
                    }, ms);
                };
                arm(this.safetyTimeoutMs);
            });
            try {
                await Promise.race([turnContext.run(turn, fn), timeout]);
            } finally {
                clearTimeout(timer);
                if (this._currentTurn === turn) this._currentTurn = null;
            }
        });
        return this._turnQueue;
    }

    /** Re-enable speech for the current turn after a barge-in cancelled it
     *  (e.g. the intro was interrupted, but the first question must still be heard). */
    _resumeTurnSpeech() {
        const turn = turnContext.getStore();
        if (turn && !turn.timedOut) turn.cancelled = false;
    }

    /** Clear processingTurn at the end of a turn — unless the safety timeout already
     *  abandoned it, in which case a newer turn may own the flag by now. */
    _endProcessing() {
        if (!turnContext.getStore()?.timedOut) this.processingTurn = false;
    }

    // ── Lifecycle ───────────────────────────────────────────────

    async start() {
        try {
            await this._startPipeline();
        } catch (err) {
            // Don't leave the Deepgram socket open when LiveKit connect/publish fails
            this.stt.stop();
            throw err;
        }
    }

    async _startPipeline() {
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

        // 3. Connect to LiveKit Room (with retry for transient failures)
        const connectTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, '[3/4] Connecting to LiveKit room...');
        const maxRetries = 3;
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                await this.room.connect(process.env.LIVEKIT_URL, token);
                agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - connectTs), attempt }, '[3/4] LiveKit room connected');
                break;
            } catch (connectErr) {
                agentLog.warn({ sessionId: this.sessionId, attempt, maxRetries, err: connectErr.message }, '[3/4] LiveKit connect failed');
                if (attempt === maxRetries) throw connectErr;
                // Exponential backoff: 1s, 2s, 4s
                const delay = 1000 * Math.pow(2, attempt - 1);
                agentLog.info({ sessionId: this.sessionId, delayMs: delay }, `[3/4] Retrying in ${delay}ms...`);
                await new Promise(r => setTimeout(r, delay));
                // Recreate room instance for fresh connection
                this.room = new Room();
                this.setupRoomEvents();
            }
        }

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

        // VAD instant barge-in — OFF by default. The raw VAD event fires on ANY
        // sound (the AI's own echo, background noise, a cough) and would cut the
        // agent off mid-sentence. Only safe with reliable echo cancellation
        // (headphones / client AEC). The transcript-confirmed barge-in below is
        // the robust default: it only fires on actual transcribed words.
        // Opt in with VAD_INSTANT_BARGE_IN=true once echo is under control.
        // STT outlives room reconnects — bind its listeners only once or transcripts get handled twice.
        if (this._sttEventsBound) return;
        this._sttEventsBound = true;
        this.stt.on("speechStarted", () => {
            if (process.env.VAD_INSTANT_BARGE_IN === 'true' && this.audioPublisher.isSpeaking) {
                agentLog.info({ sessionId: this.sessionId }, 'VAD speechStarted — instant barge-in, stopping TTS');
                this._interruptSpeech();
                this.aiStoppedSpeakingAt = Date.now();
            }
        });

        // STT transcript handler with barge-in support
        this.stt.on("transcript", async ({ transcript: text, utteranceDurationMs, fillerWordCount }) => {
            if (!this.isActive) return;

            // Barge-in: user spoke while AI was still talking
            if (this.audioPublisher.isSpeaking) {
                const wordCount = text.trim().split(/\s+/).filter(w => w).length;
                if (wordCount >= this.bargeInMinWords) {
                    agentLog.info({ sessionId: this.sessionId, transcript: text.substring(0, 80), wordCount }, 'Barge-in detected');
                    this._interruptSpeech();
                    await new Promise(r => setTimeout(r, 120));

                    const timeToAnswer = this.aiStoppedSpeakingAt > 0 ? Date.now() - this.aiStoppedSpeakingAt : 0;
                    // Queued: runs once the interrupted turn has unwound
                    await this._enqueueTurn(text, { utteranceDurationMs, fillerWordCount, timeToAnswer, bargedIn: true });
                } else {
                    agentLog.debug({ sessionId: this.sessionId, wordCount }, 'Ignored short transcript during AI speech');
                }
                return;
            }

            // Mid-turn speech is ignored, unless that turn was just interrupted (VAD barge-in)
            if (this.processingTurn && !this._currentTurn?.cancelled) return;

            const timeToAnswer = this.aiStoppedSpeakingAt > 0 ? Date.now() - this.aiStoppedSpeakingAt : 0;
            await this._enqueueTurn(text, { utteranceDurationMs, fillerWordCount, timeToAnswer, bargedIn: false });
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
                            // During AI speech, only forward audio loud enough to be real
                            // user speech. This gates out the AI's own voice echoing back
                            // through the mic (which would otherwise trip VAD speechStarted
                            // and falsely barge-in on every utterance). Tune via bargeInThreshold.
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

        agentLog.info({
            sessionId: this.sessionId,
            transcript: transcript.substring(0, 150),
            words: transcript.trim().split(/\s+/).length,
            ...acousticMeta,
        }, 'User said');

        // Notify UI
        this._emitToRoom('ai_state', { state: 'thinking', text: 'Processing...' });
        // System turns (e.g. the End button) weren't spoken by the user — don't show them as speech
        if (!acousticMeta?.system) this._emitToRoom('transcript_final', { role: 'user', text: transcript });

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
            // Don't leave the UI on "Processing…" in silence — apologise and listen again
            try {
                if (!this._turnCancelled()) {
                    this._emitToRoom('ai_state', { state: 'speaking', text: 'Speaking...' });
                    await this._speakAndEmit("Sorry, I had trouble processing that. Could you say it again?");
                }
            } catch (speakErr) {
                agentLog.error({ sessionId: this.sessionId, err: speakErr.message }, 'Turn error fallback speech failed');
            }
            this._emitToRoom('ai_state', { state: 'listening', text: 'Listening...' });
        } finally {
            this._endProcessing();
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

    /** Queue raw PCM into the audio buffer. Skips if the speech epoch advanced
     *  (barge-in) since the caller started — prevents a queued chunk from
     *  resurrecting after stop(). Does NOT wait for playout; the caller awaits
     *  audioPublisher.waitForPlayout() once after the full utterance. */
    async _playAudio(pcm, epoch) {
        if (!pcm) return;
        if (epoch !== undefined && epoch !== this._speechEpoch) return;
        await this.audioPublisher.pushPCM(pcm);
    }

    /**
     * Speak text via pipelined TTS with socket.io transcript + subtitle sync.
     * Handles all three TTS providers transparently.
     */
    async _speakAndEmit(text, emitStart = true, emitEnd = true) {
        if (!text) return;
        // A barge-in cancels the rest of this turn — e.g. the question queued after the feedback
        const turn = turnContext.getStore();
        if (turn?.cancelled) return;
        // Capture the epoch at the start. A barge-in bumps this._speechEpoch,
        // and every loop iteration below bails the instant it no longer matches.
        const myEpoch = this._speechEpoch;
        if (emitStart) this._emitToRoom('ai_speech', { action: 'start' });

        const provider = (process.env.INTERVIEW_TTS_PROVIDER || "polly").toLowerCase();

        // Speech marks for full text (subtitles)
        const marksPromise = provider === "deepgram"
            ? Promise.resolve(estimateWordTimings(text))
            : getSpeechMarks(text).catch(() => estimateWordTimings(text));

        let firstChunk = true;

        // ── Pre-roll buffer ─────────────────────────────────────────
        // The native AudioSource starts playing the INSTANT it receives the
        // first frame. The TTS first chunk is often tiny (~40ms), so if the
        // second chunk is even slightly delayed the buffer underruns and you
        // hear the voice "drop then pick up". We accumulate ~400ms of audio
        // before the first frame is queued, giving the jitter buffer a head
        // start so playback stays continuous.
        const PREROLL_SAMPLES = Math.floor(this.sampleRate * 0.4); // ~400ms lead
        let preroll = [];
        let prerollLen = 0;
        let prerollDone = false;

        const flushPreroll = async () => {
            prerollDone = true;
            if (prerollLen === 0) return;
            const combined = new Int16Array(prerollLen);
            let o = 0;
            for (const c of preroll) { combined.set(c, o); o += c.length; }
            preroll = [];
            prerollLen = 0;
            await this._playAudio(combined, myEpoch);
        };

        // Stream the FULL text in one TTS request — the provider streams chunks
        // continuously, which is smooth. Splitting per-sentence would add the
        // provider's first-chunk latency (~900ms) as a gap between sentences.
        for await (const { pcm, text: chunkText } of generatePCMPipelined(text, String(this.sampleRate))) {
            if (this._speechEpoch !== myEpoch || turn?.cancelled) break;

            if (chunkText) {
                this._emitToRoom('ai_speech', { action: 'sentence', text: chunkText });
            }

            if (firstChunk) {
                firstChunk = false;
                const marks = await marksPromise;
                if (marks.length > 0) {
                    this._emitToRoom('ai_subtitle', { words: marks, text });
                }
            }

            if (!pcm || pcm.length === 0) continue;

            if (!prerollDone) {
                preroll.push(pcm);
                prerollLen += pcm.length;
                if (prerollLen >= PREROLL_SAMPLES) await flushPreroll();
            } else {
                await this._playAudio(pcm, myEpoch);
            }
        }

        // Flush leftover pre-roll (utterance shorter than the pre-roll window).
        if (this._speechEpoch === myEpoch && !prerollDone) await flushPreroll();

        // Diagnostic: if the TTS stream yielded nothing, the agent stays silent.
        // This is the #1 cause of "agent not speaking" in production — usually a
        // missing/invalid TTS provider key or a wrong INTERVIEW_TTS_PROVIDER.
        if (firstChunk) {
            agentLog.warn(
                { sessionId: this.sessionId, provider, textPreview: text.substring(0, 60) },
                'TTS produced NO audio — check INTERVIEW_TTS_PROVIDER and its API key'
            );
        }

        // Wait for the jitter buffer to drain so we don't cut off the tail of
        // the utterance or report "done speaking" while audio is still playing.
        // Skipped on barge-in (epoch changed) — stop() already flushed the queue.
        if (this._speechEpoch === myEpoch) {
            await this.audioPublisher.waitForPlayout();
            this.aiStoppedSpeakingAt = Date.now();
        }

        if (emitEnd) this._emitToRoom('ai_speech', { action: 'end' });
    }

    // ── Socket.io helper ────────────────────────────────────────

    _emitToRoom(event, data) {
        if (this.io) this.io.to(this.sessionId).emit(event, data);
    }
}
