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
        this._queue = Promise.resolve();
        this._residual = new Int16Array(0);
    }

    /**
     * Pushes a complete raw PCM (Int16) buffer to LiveKit in 10ms AudioFrame chunks.
     * Queues calls internally to ensure sequential playback and prevent noise.
     * @param {Int16Array} pcmData - The raw PCM data to play
     */
    async pushPCM(pcmData) {
        // Queue the playback task to ensure serial execution
        this._queue = this._queue.then(() => this._internalPushPCM(pcmData));
        return this._queue;
    }

    async _internalPushPCM(pcmData) {
        if (!pcmData || (pcmData.length === 0 && this._residual.length === 0)) return;

        const isFirstInSequence = !this.isSpeaking;
        if (isFirstInSequence) {
            this.isSpeaking = true;
            this.stopFlag = false;
            this._totalChunksSent = 0;
            this._playbackStartTime = performance.now();
            agentLog.info({ samples: pcmData.length, sampleRate: this.sampleRate }, 'AudioPublisher playback start');
        }

        // Combine with residual samples from previous call
        let dataToProcess;
        if (this._residual.length > 0) {
            dataToProcess = new Int16Array(this._residual.length + pcmData.length);
            dataToProcess.set(this._residual);
            dataToProcess.set(pcmData, this._residual.length);
            this._residual = new Int16Array(0);
        } else {
            dataToProcess = pcmData;
        }

        const totalLength = dataToProcess.length;
        let offset = 0;

        // Process all complete 10ms chunks
        while (offset + this.samplesPerChunk <= totalLength) {
            if (this.stopFlag) break;

            const chunkData = dataToProcess.slice(offset, offset + this.samplesPerChunk);
            const frame = new AudioFrame(
                chunkData,
                this.sampleRate,
                this.channels,
                this.samplesPerChunk
            );

            try {
                await this.source.captureFrame(frame);
                this._totalChunksSent++;

                // Precise pacing
                const nextChunkTime = this._playbackStartTime + (this._totalChunksSent * 10);
                const now = performance.now();
                const delay = nextChunkTime - now;

                if (delay > 0) {
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            } catch (err) {
                agentLog.error({ err: err.message }, 'LiveKit captureFrame error');
                break;
            }
            offset += this.samplesPerChunk;
        }

        // Store leftovers for next call
        if (!this.stopFlag && offset < totalLength) {
            this._residual = dataToProcess.slice(offset);
        }
    }

    /** Mark playback as finished — called after turn completes or on interrupt. */
    finishPlayback() {
        // Flush any remaining residual as a padded chunk if it's significant
        if (this._residual.length > 0 && !this.stopFlag) {
            const padded = new Int16Array(this.samplesPerChunk);
            padded.set(this._residual);
            this.pushPCM(padded);
            this._residual = new Int16Array(0);
        }

        this._queue.then(() => {
            if (this.isSpeaking) {
                const playbackMs = Math.round(performance.now() - this._playbackStartTime);
                agentLog.info({ chunksSent: this._totalChunksSent, playbackMs }, 'AudioPublisher playback complete');
            }
            this.isSpeaking = false;
        });
    }

    stop() {
        this.stopFlag = true;
        this._residual = new Int16Array(0);
        if (this.isSpeaking) {
            const playbackMs = Math.round(performance.now() - this._playbackStartTime);
            agentLog.info({ chunksSent: this._totalChunksSent, playbackMs }, 'AudioPublisher playback interrupted');
        }
        this.isSpeaking = false;
    }
}
