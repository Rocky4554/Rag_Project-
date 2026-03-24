import {
    PollyClient,
    SynthesizeSpeechCommand,
} from "@aws-sdk/client-polly";
import dotenv from 'dotenv';
import { ttsLog } from '../../logger.js';
dotenv.config();

/*
 * Required Environment Variables:
 *   AWS_ACCESS_KEY_ID
 *   AWS_SECRET_ACCESS_KEY
 *   AWS_REGION          (e.g. ap-south-1)
 *
 * Optional:
 *   POLLY_VOICE_ID      (default: Joanna)
 *   POLLY_ENGINE        (default: neural)
 *   POLLY_FORMAT        (default: mp3)
 */

// ── Client created ONCE at startup (not per-request) ─────────────────────────
const client = new PollyClient({
    region: process.env.AWS_REGION || "us-east-1",
});

const MAX_CHARS = 3000; // Polly hard limit per request
const MAX_RETRIES = 3;

// ── Chunk text by character limit (simple, reliable for long summaries) ───────
function chunkText(text, size = MAX_CHARS) {
    // Split on sentence boundaries, then group into chunks under `size` chars
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];
    let cur = "";
    for (const s of sentences) {
        if ((cur + s).length > size) {
            if (cur) chunks.push(cur.trim());
            cur = s;
        } else {
            cur += (cur ? " " : "") + s;
        }
    }
    if (cur) chunks.push(cur.trim());
    return chunks;
}

// ── Convert a Node.js Readable / async iterable stream → Buffer ──────────────
async function streamToBuffer(stream) {
    const chunks = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
}

// ── Single Polly synthesis call with exponential backoff retry ────────────────
async function synthesizeChunk(text, attempt = 0) {
    try {
        const command = new SynthesizeSpeechCommand({
            Text: text,
            OutputFormat: process.env.POLLY_FORMAT || "mp3",
            VoiceId: process.env.POLLY_VOICE_ID || "Joanna",
            Engine: process.env.POLLY_ENGINE || "neural",
        });

        const response = await client.send(command);
        return await streamToBuffer(response.AudioStream);

    } catch (err) {
        if (attempt < MAX_RETRIES) {
            const delay = 500 * (attempt + 1); // 500ms, 1000ms, 1500ms
            ttsLog.warn({ attempt: attempt + 1, delay, err: err.message }, 'Polly chunk failed, retrying');
            await new Promise(r => setTimeout(r, delay));
            return synthesizeChunk(text, attempt + 1);
        }
        throw new Error(`[Polly] Failed after ${MAX_RETRIES} retries: ${err.message}`);
    }
}

// ── Main exported function ─────────────────────────────────────────────────────
export async function textToAudio(text) {
    if (!text || !text.trim()) {
        throw new Error("[Polly] textToAudio requires non-empty text");
    }

    const totalStart = performance.now();
    const chunks = chunkText(text).filter(c => c.trim());
    const voice = process.env.POLLY_VOICE_ID || "Joanna";
    ttsLog.info({ chunks: chunks.length, voice, textLength: text.length }, 'Polly synthesis started');

    const audioBuffers = [];
    for (let i = 0; i < chunks.length; i++) {
        const chunkStart = performance.now();
        const buffer = await synthesizeChunk(chunks[i]);
        audioBuffers.push(buffer);
        const chunkMs = Math.round(performance.now() - chunkStart);
        ttsLog.debug({ chunk: i + 1, total: chunks.length, durationMs: chunkMs, chunkChars: chunks[i].length }, 'Polly chunk synthesized');
    }

    const combined = Buffer.concat(audioBuffers);
    const totalMs = Math.round(performance.now() - totalStart);
    ttsLog.info({ totalMs, chunks: chunks.length, audioBytes: combined.length }, 'Polly synthesis complete');

    return combined.toString("base64");
}
