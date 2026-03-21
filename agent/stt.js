import WebSocket from "ws";
import { EventEmitter } from "events";
import dotenv from "dotenv";

dotenv.config();

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
        this.endpointing = parseInt(process.env.DEEPGRAM_STT_ENDPOINTING_MS || "500", 10);

        this.currentTranscript = "";
    }

    start() {
        this._stopped = false;
        this._connect();
    }

    _connect() {
        if (this._stopped) return;

        const apiKey = process.env.DEEPGRAM_API_KEY;
        if (!apiKey) throw new Error("DEEPGRAM_API_KEY is missing");

        const params = new URLSearchParams({
            encoding: "linear16",
            sample_rate: "48000",   // LiveKit track output is 48kHz by default
            channels: "1",
            model: this.model,
            language: this.language,
            smart_format: String(this.smartFormat),
            interim_results: "true",
            endpointing: String(this.endpointing),
        });

        const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

        console.log(`[STT] Connecting to Deepgram...`);

        this.socket = new WebSocket(url, {
            headers: {
                Authorization: `Token ${apiKey}`
            }
        });

        this.socket.on("open", () => {
            console.log("[STT] Deepgram WebSocket opened.");
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
                            this.currentTranscript += text + " ";
                        }
                    }

                    if (msg.speech_final) {
                        const finalAnswer = this.currentTranscript.trim();
                        if (finalAnswer.length > 0) {
                            console.log(`[STT] Speech Final: "${finalAnswer}"`);
                            this.emit("transcript", finalAnswer);
                        }
                        this.currentTranscript = ""; // reset for next utterance
                    }
                }
            } catch (err) {
                console.error("[STT] Parse error on incoming message:", err);
            }
        });

        this.socket.on("close", () => {
            console.log("[STT] Deepgram WebSocket closed.");
            this.isReady = false;
            this._stopKeepalive();

            // Auto-reconnect unless explicitly stopped
            if (!this._stopped) {
                console.log("[STT] Reconnecting in 1s...");
                setTimeout(() => this._connect(), 1000);
            }
        });

        this.socket.on("error", (err) => {
            console.error("[STT] Deepgram WebSocket error:", err.message);
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
            // Use byteOffset/byteLength to send only the viewed slice,
            // not the entire underlying ArrayBuffer (which may be larger).
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