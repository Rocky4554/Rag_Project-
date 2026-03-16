import {
    PollyClient,
    SynthesizeSpeechCommand,
} from "@aws-sdk/client-polly";
import dotenv from "dotenv";
dotenv.config();

const client = new PollyClient({
    region: process.env.AWS_REGION || "us-east-1",
});

/**
 * Convert a Node.js Readable / async iterable stream to a raw Buffer
 */
async function streamToBuffer(stream) {
    const chunks = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
}

/**
 * Converts text directly to raw PCM audio format required by LiveKit.
 * Output: 16kHz or 48kHz, 16-bit, mono PCM.
 */
export async function generatePCM(text, sampleRate = "16000") {
    if (!text || !text.trim()) return null;

    try {
        console.log(`[Agent TTS] Requesting PCM audio for text: "${text.substring(0, 50)}..."`);
        
        const command = new SynthesizeSpeechCommand({
            Text: text,
            OutputFormat: "pcm", // CRITICAL: LiveKit needs uncompressed PCM, not MP3
            SampleRate: sampleRate, // "16000" or "48000" (LiveKit defaults to 48k but handles 16k well)
            VoiceId: process.env.POLLY_VOICE_ID || "Joanna",
            Engine: process.env.POLLY_ENGINE || "neural",
        });

        const response = await client.send(command);
        const buffer = await streamToBuffer(response.AudioStream);
        
        // Convert Node Buffer to Int16Array
        // PCM 16-bit means 2 bytes per sample
        const int16Array = new Int16Array(
            buffer.buffer, 
            buffer.byteOffset, 
            buffer.length / Int16Array.BYTES_PER_ELEMENT
        );
        
        console.log(`[Agent TTS] PCM Generated. Samples: ${int16Array.length}`);
        return int16Array;

    } catch (err) {
        console.error("[Agent TTS] Failed to generate PCM audio:", err);
        return null;
    }
}
