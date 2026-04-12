import { AudioSource, AudioFrame } from "@livekit/rtc-node";
import { agentLog } from "../../lib/logger.js";

// LiveKit requires PCM audio pushed as AudioFrame objects in 10ms chunks
export class AudioPublisher {
    constructor(sampleRate = 16000, channels = 1) {
        this.sampleRate = sampleRate;
        this.channels = channels;
        // 10ms worth of samples per channel: sampleRate * 10 / 1000
        this.samplesPerChunk = Math.floor((sampleRate * 10) / 1000);
        this.source = new AudioSource(sampleRate, channels);
        this.isSpeaking = false;
        this.stopFlag = false;
        this._totalChunksSent = 0;
        this._playbackStartTime = 0;
    }

    /**
     * Pushes a complete raw PCM (Int16) buffer to LiveKit in 10ms AudioFrame chunks.
     * Optimized: minimal logging per-call, batch metrics only on first/last chunk.
     * @param {Int16Array} pcmData - The raw PCM data to play
     */
    async pushPCM(pcmData) {
        const isFirstChunk = !this.isSpeaking;
        if (isFirstChunk) {
            this.isSpeaking = true;
            this.stopFlag = false;
            this._totalChunksSent = 0;
            this._playbackStartTime = performance.now();
            agentLog.info({ samples: pcmData.length, sampleRate: this.sampleRate }, 'AudioPublisher playback start');
        }

        for (let offset = 0; offset < pcmData.length; offset += this.samplesPerChunk) {
            if (this.stopFlag) break;

            let chunkData;
            if (offset + this.samplesPerChunk <= pcmData.length) {
                // MUST use .slice() not .subarray() — subarray shares the parent
                // ArrayBuffer, and livekit-rtc-node's AudioFrame.protoInfo() reads
                // from this.data.buffer (byte 0 of the underlying buffer).  With
                // subarray every frame would contain the same first 160 samples.
                chunkData = pcmData.slice(offset, offset + this.samplesPerChunk);
            } else {
                chunkData = new Int16Array(this.samplesPerChunk);
                chunkData.set(pcmData.subarray(offset));
            }

            const frame = new AudioFrame(
                chunkData,
                this.sampleRate,
                this.channels,
                this.samplesPerChunk
            );

            await this.source.captureFrame(frame);
            this._totalChunksSent++;

            // Precise pacing: calculate when the next chunk SHOULD be sent
            // and wait exactly that long, avoiding event loop drift
            const nextChunkTime = this._playbackStartTime + (this._totalChunksSent * 10);
            const now = performance.now();
            const delay = nextChunkTime - now;

            if (delay > 0) {
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    /** Mark playback as finished — called after turn completes or on interrupt. */
    finishPlayback() {
        if (this.isSpeaking) {
            const playbackMs = Math.round(performance.now() - this._playbackStartTime);
            agentLog.info({ chunksSent: this._totalChunksSent, playbackMs }, 'AudioPublisher playback complete');
        }
        this.isSpeaking = false;
    }

    stop() {
        this.stopFlag = true;
        if (this.isSpeaking) {
            const playbackMs = Math.round(performance.now() - this._playbackStartTime);
            agentLog.info({ chunksSent: this._totalChunksSent, playbackMs }, 'AudioPublisher playback interrupted');
        }
        this.isSpeaking = false;
    }
}
