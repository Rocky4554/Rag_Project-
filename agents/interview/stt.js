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

        // Incomplete-transcript guard: if speech_final fires mid-clause (e.g. "can you",
        // "and the", "because"), hold for 2s before emitting in case the user continues.
        this._pendingFinalTimer = null;
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

        const useFlux = process.env.DEEPGRAM_USE_FLUX === 'true';
        const model = useFlux ? 'flux' : this.model;

        const baseParams = {
            encoding: "linear16",
            sample_rate: sampleRate,
            channels: "1",
            model,
            language: this.language,
            smart_format: String(this.smartFormat),
            filler_words: "true",
            interim_results: "true",
            vad_events: "true",
            punctuate: "true",
        };

        if (useFlux) {
            baseParams.eot_threshold = process.env.DEEPGRAM_FLUX_EOT_THRESHOLD || "0.7";
        } else {
            baseParams.endpointing = String(this.endpointing);
            baseParams.utterance_end_ms = process.env.DEEPGRAM_STT_UTTERANCE_END_MS || "1500";
        }

        const params = new URLSearchParams(baseParams);

        const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

        agentLog.info({ model, endpointingMs: useFlux ? null : this.endpointing, flux: useFlux }, 'STT connecting to Deepgram');
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
                            // New speech arrived — cancel any pending hold timer
                            this._clearPendingFinal();

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
                        if (this.currentTranscript.trim().length > 0 && this._looksIncomplete(this.currentTranscript)) {
                            agentLog.info(
                                { snippet: this.currentTranscript.trim().slice(-40) },
                                'STT speech_final: incomplete clause detected, holding 2s'
                            );
                            this._schedulePendingFinal();
                        } else {
                            this._clearPendingFinal();
                            this._emitFinal('speech_final');
                        }
                    }
                }

                // UtteranceEnd: absolute safety net — fires after utterance_end_ms of
                // silence. Always emits whatever is buffered, cancelling any held timer.
                if (msg.type === "UtteranceEnd") {
                    if (this.currentTranscript.trim().length > 0) {
                        this._clearPendingFinal();
                        agentLog.info('STT UtteranceEnd fallback — flushing buffered transcript');
                        this._emitFinal('utterance_end');
                    }
                }

                if (msg.type === "SpeechStarted") {
                    this.emit("speechStarted");
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
     * Returns true when the transcript tail is a dangling clause — conjunction,
     * preposition, article, or question fragment — with no terminal punctuation.
     * High-precision check: only flags clear mid-sentence cut-offs.
     */
    _looksIncomplete(text) {
        const t = text.trim().toLowerCase().replace(/[,;:]+$/, '');
        // smart_format adds '.', '?', '!' to complete sentences
        if (/[.?!]$/.test(t)) return false;
        // Words that almost never end a complete thought
        const dangling = /\b(and|but|or|nor|so|yet|because|although|though|since|unless|while|if|when|the|a|an|this|that|these|those|to|for|with|on|in|at|of|from|by|can you|could you|would you|will you|how do|how does|what is|what are|where is)\s*$/;
        return dangling.test(t);
    }

    _schedulePendingFinal() {
        this._clearPendingFinal();
        this._pendingFinalTimer = setTimeout(() => {
            this._pendingFinalTimer = null;
            if (this.currentTranscript.trim().length > 0) {
                agentLog.info('STT delayed emit after incomplete-clause hold');
                this._emitFinal('speech_final_delayed');
            }
        }, 2000);
    }

    _clearPendingFinal() {
        if (this._pendingFinalTimer) {
            clearTimeout(this._pendingFinalTimer);
            this._pendingFinalTimer = null;
        }
    }

    /**
     * Emit the accumulated transcript as a final result and reset state.
     */
    _emitFinal(source) {
        const finalAnswer = this.currentTranscript.trim();
        if (finalAnswer.length === 0) return;

        const utteranceDurationMs = this._utteranceStartMs
            ? Date.now() - this._utteranceStartMs
            : 0;

        agentLog.info({
            transcript: finalAnswer.substring(0, 100),
            words: finalAnswer.trim().split(/\s+/).length,
            utteranceDurationMs,
            fillerWordCount: this._fillerWordCount,
            source,
        }, 'STT speech final');

        this.emit("transcript", {
            transcript: finalAnswer,
            utteranceDurationMs,
            fillerWordCount: this._fillerWordCount,
        });

        // Reset for next utterance
        this.currentTranscript = "";
        this._utteranceStartMs = 0;
        this._fillerWordCount = 0;
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
        this._clearPendingFinal();
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        this.isReady = false;
    }
}
