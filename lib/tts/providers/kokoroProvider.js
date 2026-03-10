import dotenv from 'dotenv';
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
            console.log(`[Kokoro] Chunk ${index + 1} (${chunk.length} chars)...`);
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
            console.log(`[Kokoro] Chunk ${index + 1} done.`);
            return Buffer.from(arrayBuffer);
        } catch (err) {
            console.error(`[Kokoro] Chunk ${index + 1} failed: ${err.message}`);
            retries--;
            if (retries === 0) throw new Error(`[Kokoro] Chunk ${index + 1} failed after ${MAX_RETRIES} retries: ${err.message}`);
            console.log(`[Kokoro] Retrying chunk ${index + 1}... (${retries} left). Waiting 2s...`);
            await new Promise(res => setTimeout(res, 2000));
        }
    }
}

export async function textToAudio(text) {
    if (!text || !text.trim()) {
        throw new Error("[Kokoro] textToAudio requires non-empty text");
    }

    const apiUrl = process.env.KOKORO_API_URL || "http://localhost:8880";
    const chunks = splitBySentence(text, MAX_CHARS).filter(c => c.trim());
    console.log(`[Kokoro] ${chunks.length} chunk(s) → ${apiUrl}`);
    console.time("[Kokoro] Total TTS");

    const audioBuffers = [];
    for (let i = 0; i < chunks.length; i++) {
        audioBuffers.push(await fetchChunk(apiUrl, chunks[i], i));
    }

    console.timeEnd("[Kokoro] Total TTS");
    console.log("[Kokoro] Merging buffers...");
    const combined = Buffer.concat(audioBuffers);
    console.log("[Kokoro] Merging complete. Ready to play!");
    return combined.toString("base64");
}
