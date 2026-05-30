import { AudioSource, AudioFrame } from "@livekit/rtc-node";
import { agentLog } from "../../lib/logger.js";

// LiveKit requires PCM audio pushed as AudioFrame objects in 10ms chunks.
//
// The native AudioSource keeps an internal jitter buffer (queueSizeMs). We feed
// frames into it as fast as captureFrame accepts them (captureFrame applies
// backpressure once the buffer is full), and let the native side handle the
// real-time playback pacing. This is what makes streamed-TTS playback smooth:
// the buffer absorbs the network gaps between chunks so there are no dropouts.
//
// IMPORTANT: do NOT add manual setTimeout pacing here. Doing so keeps the native
// buffer near-empty, so every inter-chunk network gap becomes an audible break
// ("robotic / connection-dropping" speech). That was the old bug.
const QUEUE_SIZE_MS = 1000;

export class AudioPublisher {
    constructor(sampleRate = 16000, channels = 1) {
        this.sampleRate = sampleRate;
        this.channels = channels;
        // 10ms worth of samples per channel: sampleRate * 10 / 1000
        this.samplesPerChunk = Math.floor((sampleRate * 10) / 1000);
        this.source = new AudioSource(sampleRate, channels, QUEUE_SIZE_MS);
        this.isSpeaking = false;
        this.stopFlag = false;
    }

    /**
     * Queue a complete raw PCM (Int16) buffer for playback in 10ms AudioFrames.
     * Returns once all frames are accepted into the native buffer (NOT once they
     * have finished playing). Call waitForPlayout() afterwards to await real end.
     * @param {Int16Array} pcmData - The raw PCM data to play
     */
    async pushPCM(pcmData) {
        if (!pcmData || pcmData.length === 0) return;
        this.isSpeaking = true;
        this.stopFlag = false;

        const startTime = performance.now();
        let chunksSent = 0;

        for (let offset = 0; offset < pcmData.length; offset += this.samplesPerChunk) {
            if (this.stopFlag) {
                agentLog.debug('AudioPublisher playback interrupted');
                break;
            }

            let chunkData;
            if (offset + this.samplesPerChunk <= pcmData.length) {
                // MUST use .slice() not .subarray() — subarray shares the parent
                // ArrayBuffer, and livekit-rtc-node's AudioFrame.protoInfo() reads
                // from this.data.buffer (byte 0 of the underlying buffer). With
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

            // captureFrame backpressures on the native buffer — no manual pacing.
            try {
                await this.source.captureFrame(frame);
            } catch {
                // Source closed or queue cleared (barge-in) — stop feeding.
                break;
            }
            chunksSent++;
        }

        agentLog.debug({ chunksSent, queuedMs: Math.round(performance.now() - startTime) }, 'AudioPublisher chunk queued');
    }

    /**
     * Wait for the native buffer to fully drain (i.e. audio actually finished
     * playing), then mark speech as done. Safe to call when nothing is queued.
     */
    async waitForPlayout() {
        try {
            await this.source.waitForPlayout();
        } catch {
            /* source may be closed — ignore */
        }
        this.isSpeaking = false;
    }

    /** Instantly drop all buffered audio (barge-in) and stop feeding. */
    stop() {
        this.stopFlag = true;
        this.isSpeaking = false;
        try {
            // Flush the ~1s of buffered audio so an interrupt cuts off
            // immediately instead of playing out the remaining queue.
            this.source.clearQueue();
        } catch {
            /* older rtc-node without clearQueue — ignore */
        }
    }
}
