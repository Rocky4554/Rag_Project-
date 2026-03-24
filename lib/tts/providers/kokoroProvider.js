import dotenv from 'dotenv';
import { ttsLog } from '../../logger.js';
dotenv.config();

/*
 * Legacy Kokoro-FastAPI TTS provider.
 * Requires a running Kokoro Docker container.
 *
 * Required Environment Variables:
 *   KOKORO_API_URL  (default: http://localhost:8880)
 */

const MAX_CHARS = 700;
const MAX_RETRIES = 3;
const CONCURRENCY = 1; // CPU mode: sequential is safest

function splitBySentence(text, max) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];
    let cur = "";
    for (const s of sentences) {
        if ((cur + s).length > max) {
            if (cur) chunks.push(cur.trim());
            cur = s;
        } else {
            cur += (cur ? " " : "") + s;
        }
    }
    if (cur) chunks.push(cur.trim());
    return chunks;
}

async function fetchChunk(apiUrl, chunk, index) {
    let retries = MAX_RETRIES;
    while (retries > 0) {
        try {
            ttsLog.debug({ chunk: index + 1, chars: chunk.length }, 'Kokoro chunk started');
            const response = await fetch(`${apiUrl}/v1/audio/speech`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: "kokoro",
                    input: chunk,
                    voice: process.env.KOKORO_VOICE || "af_heart",
                    response_format: "mp3"
                })
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`TTS API Error (${response.status}): ${errText}`);
            }

            const arrayBuffer = await response.arrayBuffer();
            ttsLog.debug({ chunk: index + 1 }, 'Kokoro chunk done');
            return Buffer.from(arrayBuffer);
        } catch (err) {
            ttsLog.warn({ chunk: index + 1, err: err.message, retriesLeft: retries - 1 }, 'Kokoro chunk failed');
            retries--;
            if (retries === 0) throw new Error(`[Kokoro] Chunk ${index + 1} failed after ${MAX_RETRIES} retries: ${err.message}`);
            await new Promise(res => setTimeout(res, 2000));
        }
    }
}

export async function textToAudio(text) {
    if (!text || !text.trim()) {
        throw new Error("[Kokoro] textToAudio requires non-empty text");
    }

    const totalStart = performance.now();
    const apiUrl = process.env.KOKORO_API_URL || "http://localhost:8880";
    const chunks = splitBySentence(text, MAX_CHARS).filter(c => c.trim());
    ttsLog.info({ chunks: chunks.length, apiUrl, textLength: text.length }, 'Kokoro synthesis started');

    const audioBuffers = [];
    for (let i = 0; i < chunks.length; i++) {
        audioBuffers.push(await fetchChunk(apiUrl, chunks[i], i));
    }

    const combined = Buffer.concat(audioBuffers);
    const totalMs = Math.round(performance.now() - totalStart);
    ttsLog.info({ totalMs, chunks: chunks.length, audioBytes: combined.length }, 'Kokoro synthesis complete');

    return combined.toString("base64");
}
