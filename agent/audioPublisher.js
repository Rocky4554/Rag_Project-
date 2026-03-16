import { AudioSource, AudioFrame } from "@livekit/rtc-node";

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
    }

    /**
     * Pushes a complete raw PCM (Int16) buffer to LiveKit in 10ms AudioFrame chunks.
     * @param {Int16Array} pcmData - The raw PCM data to play
     */
    async pushPCM(pcmData) {
        this.isSpeaking = true;
        this.stopFlag = false;
        console.log(`[AudioPublisher] Starting playback of ${pcmData.length} samples at ${this.sampleRate}Hz`);

        for (let offset = 0; offset < pcmData.length; offset += this.samplesPerChunk) {
            if (this.stopFlag) {
                console.log("[AudioPublisher] Playback interrupted.");
                break;
            }

            // Slice exactly samplesPerChunk samples (pad last chunk if needed)
            let chunkData;
            if (offset + this.samplesPerChunk <= pcmData.length) {
                chunkData = pcmData.subarray(offset, offset + this.samplesPerChunk);
            } else {
                chunkData = new Int16Array(this.samplesPerChunk);
                chunkData.set(pcmData.subarray(offset));
            }

            // Wrap in LiveKit AudioFrame — captureFrame requires this, not raw Int16Array
            const frame = new AudioFrame(
                chunkData,
                this.sampleRate,
                this.channels,
                this.samplesPerChunk  // samplesPerChannel
            );

            await this.source.captureFrame(frame);
        }

        console.log("[AudioPublisher] Playback complete.");
        this.isSpeaking = false;
    }

    stop() {
        this.stopFlag = true;
        this.isSpeaking = false;
    }
}