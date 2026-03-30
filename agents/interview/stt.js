import WebSocket from "ws";
import { EventEmitter } from "events";
import dotenv from "dotenv";
import { agentLog } from "../../lib/logger.js";

dotenv.config();

// Filler words to detect and count (used for behavioral context injection)
const FILLER_WORD_REGEX = /\b(uh|um|er|erm|hmm|hm)\b/gi;

export class DeepgramSTT extends EventEmitter {
    constructor() {
        super();
        this.socket = null;
        this.isReady = false;
        this._stopped = false;
        this._keepaliveTimer = null;

        // Settings from .env
        this.model = process.env.DEEPGRAM_STT_MODEL || "nova-3";
        this.language = process.env.DEEPGRAM_STT_LANGUAGE || "en";
        this.smartFormat = (process.env.DEEPGRAM_STT_SMART_FORMAT || "true") === "true";

        // Improvement #2: 700ms endpointing gives candidates time to pause mid-thought
        // without the agent cutting them off. 300ms was too aggressive for technical answers
        // where candidates pause to recall terminology or construct complex sentences.
        this.endpointing = parseInt(process.env.DEEPGRAM_STT_ENDPOINTING_MS || "700", 10);

        this.currentTranscript = "";

        // Improvement #4: Acoustic metadata — per-utterance timing + filler tracking
        this._utteranceStartMs = 0;   // timestamp of first final chunk for this utterance
        this._fillerWordCount = 0;     // running count of uh/um/er in current utterance
    }

    start() {
        this._stopped = false;
        this._connect();
    }

    _connect() {
        if (this._stopped) return;

        const apiKey = process.env.DEEPGRAM_API_KEY;
        if (!apiKey) throw new Error("DEEPGRAM_API_KEY is missing");

        const sampleRate = process.env.DEEPGRAM_STT_SAMPLE_RATE || "48000";

        const params = new URLSearchParams({
            encoding: "linear16",
            sample_rate: sampleRate,
            channels: "1",
            model: this.model,
            language: this.language,
            smart_format: String(this.smartFormat),
            filler_words: "true",
            interim_results: "true",
            endpointing: String(this.endpointing),
            // utterance_end_ms: extra silence buffer after speech_final detection
            // Gives more time for natural pauses in longer answers
            utterance_end_ms: process.env.DEEPGRAM_STT_UTTERANCE_END_MS || "1500",
        });

        const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

        agentLog.info({ model: this.model, endpointingMs: this.endpointing }, 'STT connecting to Deepgram');
        this._connectTs = Date.now();

        this.socket = new WebSocket(url, {
            headers: {
                Authorization: `Token ${apiKey}`
            }
        });

        this.socket.on("open", () => {
            agentLog.info({ ms: Math.round(Date.now() - (this._connectTs || Date.now())), model: this.model }, 'STT Deepgram WebSocket opened');
            this.isReady = true;
            this._startKeepalive();
        });

        this.socket.on("message", (data) => {
            try {
                const msg = JSON.parse(data);
                if (msg.type === "Results" && msg.channel && msg.channel.alternatives[0]) {
                    const alt = msg.channel.alternatives[0];
                    const text = alt.transcript;

                    if (text) {
                        if (msg.is_final) {
                            // Track utterance start on first final chunk
                            if (!this._utteranceStartMs) {
                                this._utteranceStartMs = Date.now();
                            }
                            this.currentTranscript += text + " ";

                            // Count filler words from each final chunk as they arrive
                            const fillers = text.match(FILLER_WORD_REGEX);
                            if (fillers) this._fillerWordCount += fillers.length;
                        }
                    }

                    if (msg.speech_final) {
                        const finalAnswer = this.currentTranscript.trim();
                        if (finalAnswer.length > 0) {
                            const utteranceDurationMs = this._utteranceStartMs
                                ? Date.now() - this._utteranceStartMs
                                : 0;

                            agentLog.info({
                                transcript: finalAnswer.substring(0, 100),
                                words: finalAnswer.trim().split(/\s+/).length,
                                utteranceDurationMs,
                                fillerWordCount: this._fillerWordCount,
                            }, 'STT speech final');

                            // Emit an object (not just a string) so worker gets acoustic metadata
                            this.emit("transcript", {
                                transcript: finalAnswer,
                                utteranceDurationMs,
                                fillerWordCount: this._fillerWordCount,
                            });
                        }
                        // Reset for next utterance
                        this.currentTranscript = "";
                        this._utteranceStartMs = 0;
                        this._fillerWordCount = 0;
                    }
                }
            } catch (err) {
                agentLog.error({ err: err.message }, 'STT parse error');
            }
        });

        this.socket.on("close", () => {
            agentLog.info('STT Deepgram WebSocket closed');
            this.isReady = false;
            this._stopKeepalive();

            // Auto-reconnect unless explicitly stopped
            if (!this._stopped) {
                agentLog.info('STT reconnecting in 1s');
                setTimeout(() => this._connect(), 1000);
            }
        });

        this.socket.on("error", (err) => {
            agentLog.error({ err: err.message }, 'STT Deepgram WebSocket error');
            this.isReady = false;
        });
    }

    /**
     * Send a keepalive message every 8s to prevent Deepgram from closing
     * the WebSocket during AI speaking pauses (its idle timeout is ~10-12s).
     */
    _startKeepalive() {
        this._stopKeepalive();
        this._keepaliveTimer = setInterval(() => {
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.socket.send(JSON.stringify({ type: "KeepAlive" }));
            }
        }, 8000);
    }

    _stopKeepalive() {
        if (this._keepaliveTimer) {
            clearInterval(this._keepaliveTimer);
            this._keepaliveTimer = null;
        }
    }

    /**
     * Pushes raw audio frames (Int16Array) from LiveKit directly into Deepgram
     */
    pushAudio(int16Array) {
        if (this.isReady && this.socket && this.socket.readyState === WebSocket.OPEN) {
            const buf = Buffer.from(
                int16Array.buffer,
                int16Array.byteOffset,
                int16Array.byteLength
            );
            this.socket.send(buf);
        }
    }

    stop() {
        this._stopped = true;
        this._stopKeepalive();
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        this.isReady = false;
    }
}
