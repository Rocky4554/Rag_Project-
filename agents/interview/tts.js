/**
 * Multi-provider TTS for the interview agent.
 *
 * Providers:
 *   - polly     (AWS Polly)      — request/response, fake-streamed per sentence
 *   - deepgram  (Deepgram Aura)  — real streaming PCM
 *   - elevenlabs (ElevenLabs)    — real streaming PCM
 *
 * All providers expose the same interface:
 *   generatePCM(text, sampleRate)            → Promise<Int16Array|null>
 *   generatePCMPipelined(text, sampleRate)   → AsyncGenerator<{ pcm, index, text }>
 *
 * The active provider is selected via the INTERVIEW_TTS_PROVIDER env var
 * (default: "polly").  Switch at deploy time — no code changes needed.
 */
import {
    PollyClient,
    SynthesizeSpeechCommand,
} from "@aws-sdk/client-polly";
import { DeepgramClient } from "@deepgram/sdk";
import dotenv from "dotenv";
import { ttsLog } from "../../lib/logger.js";

dotenv.config();

// ═══════════════════════════════════════════════════════════════════
// SHARED UTILS
// ═══════════════════════════════════════════════════════════════════

/** Convert a Node.js Readable / async iterable stream to a raw Buffer. */
async function streamToBuffer(stream) {
    const chunks = [];
    for await (const chunk of stream) {
        chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
}

/** Split text into sentences for pipelined TTS. */
export function splitIntoSentences(text) {
    if (!text) return [];
    const sentences = text.match(/[^.!?]+[.!?]+[\s]?|[^.!?]+$/g);
    if (!sentences) return [text.trim()];
    return sentences.map(s => s.trim()).filter(s => s.length > 0);
}

/** Convert a raw Buffer of linear16 PCM bytes to an Int16Array. */
function bufferToInt16(buffer) {
    return new Int16Array(
        buffer.buffer,
        buffer.byteOffset,
        buffer.length / Int16Array.BYTES_PER_ELEMENT
    );
}

// ═══════════════════════════════════════════════════════════════════
// PROVIDER: AWS POLLY  (request/response → fake-streamed per sentence)
// ═══════════════════════════════════════════════════════════════════

const pollyClient = new PollyClient({
    region: process.env.AWS_REGION || "us-east-1",
});

/**
 * Get word-level timestamps from Polly's SpeechMarks API.
 * Returns: [{ time: 0, value: "Hello" }, { time: 400, value: "how" }, ...]
 * time is in milliseconds from audio start.
 */
async function pollyGetSpeechMarks(text) {
    if (!text?.trim()) return [];
    const voiceId = process.env.POLLY_VOICE_ID || "Joanna";
    const engine = process.env.POLLY_ENGINE || "neural";

    try {
        const command = new SynthesizeSpeechCommand({
            Text: text,
            OutputFormat: "json",
            SampleRate: "16000",
            VoiceId: voiceId,
            Engine: engine,
            SpeechMarkTypes: ["word"],
        });
        const response = await pollyClient.send(command);
        const raw = await streamToBuffer(response.AudioStream);
        // Polly returns newline-delimited JSON objects
        const marks = raw.toString("utf-8")
            .split("\n")
            .filter(line => line.trim())
            .map(line => JSON.parse(line))
            .filter(m => m.type === "word")
            .map(m => ({ time: m.time, value: m.value }));
        return marks;
    } catch (err) {
        ttsLog.warn({ provider: "polly", err: err.message }, "SpeechMarks failed, using estimation");
        return estimateWordTimings(text);
    }
}

/**
 * Estimate word timings when speech marks are unavailable (Deepgram, ElevenLabs).
 * Uses average speaking rate of ~150 words/minute = ~400ms per word.
 */
export function estimateWordTimings(text, totalDurationMs = null) {
    if (!text?.trim()) return [];
    const words = text.trim().split(/\s+/);
    const avgMsPerWord = totalDurationMs ? totalDurationMs / words.length : 400;
    return words.map((word, i) => ({
        time: Math.round(i * avgMsPerWord),
        value: word,
    }));
}

async function pollyGeneratePCM(text, sampleRate = "16000") {
    if (!text?.trim()) return null;
    const start = performance.now();
    const voiceId = process.env.POLLY_VOICE_ID || "Joanna";
    const engine = process.env.POLLY_ENGINE || "neural";

    try {
        ttsLog.info({ provider: "polly", textLen: text.length, voice: voiceId }, "TTS request");
        const command = new SynthesizeSpeechCommand({
            Text: text,
            OutputFormat: "pcm",
            SampleRate: sampleRate,
            VoiceId: voiceId,
            Engine: engine,
        });
        const response = await pollyClient.send(command);
        const buffer = await streamToBuffer(response.AudioStream);
        const pcm = bufferToInt16(buffer);
        ttsLog.info({ provider: "polly", samples: pcm.length, ms: Math.round(performance.now() - start) }, "TTS ready");
        return pcm;
    } catch (err) {
        ttsLog.error({ provider: "polly", err: err.message, ms: Math.round(performance.now() - start) }, "TTS failed");
        return null;
    }
}

/**
 * Polly fake-streaming: fire all sentence TTS concurrently,
 * yield each sentence's PCM in sequential order.
 */
async function* pollyGeneratePCMPipelined(text, sampleRate = "16000") {
    const sentences = splitIntoSentences(text);
    if (sentences.length === 0) return;

    if (sentences.length === 1) {
        const pcm = await pollyGeneratePCM(sentences[0], sampleRate);
        if (pcm) yield { pcm, index: 0, text: sentences[0] };
        return;
    }

    ttsLog.info({ provider: "polly", sentences: sentences.length }, "TTS pipeline start");

    // Fire all requests concurrently
    const promises = sentences.map((s, i) =>
        pollyGeneratePCM(s, sampleRate).then(pcm => ({ pcm, index: i }))
    );

    // Yield in order using per-index resolvers
    const results = new Array(sentences.length);
    const resolvers = new Array(sentences.length);
    const readyPromises = sentences.map((_, i) =>
        new Promise(resolve => { resolvers[i] = resolve; })
    );
    for (const p of promises) {
        p.then(r => { results[r.index] = r; resolvers[r.index](); });
    }
    for (let i = 0; i < sentences.length; i++) {
        await readyPromises[i];
        if (results[i]?.pcm) yield { ...results[i], text: sentences[i] };
    }
}

// ═══════════════════════════════════════════════════════════════════
// PROVIDER: DEEPGRAM  (TRUE streaming — chunks yielded as they arrive)
// ═══════════════════════════════════════════════════════════════════

let _dgClient = null;
function getDeepgramTTSClient() {
    if (!_dgClient) {
        const apiKey = process.env.DEEPGRAM_API_KEY;
        if (!apiKey) throw new Error("DEEPGRAM_API_KEY is required for Deepgram TTS");
        _dgClient = new DeepgramClient(apiKey);
    }
    return _dgClient;
}

/** Single-shot: send full text, collect all audio, return Int16Array. Used for short phrases. */
async function deepgramGeneratePCM(text, sampleRate = "16000") {
    if (!text?.trim()) return null;
    const start = performance.now();
    const model = process.env.DEEPGRAM_TTS_MODEL || "aura-2-en";

    try {
        ttsLog.info({ provider: "deepgram", textLen: text.length, model }, "TTS request");
        const dg = getDeepgramTTSClient();
        const response = await dg.speak.request(
            { text },
            {
                model,
                encoding: "linear16",
                sample_rate: parseInt(sampleRate),
                container: "none",
            }
        );
        const stream = await response.getStream();
        if (!stream) {
            ttsLog.error({ provider: "deepgram" }, "TTS returned no stream");
            return null;
        }
        const buffer = await streamToBuffer(stream);
        const pcm = bufferToInt16(buffer);
        ttsLog.info({ provider: "deepgram", samples: pcm.length, ms: Math.round(performance.now() - start) }, "TTS ready");
        return pcm;
    } catch (err) {
        ttsLog.error({ provider: "deepgram", err: err.message, ms: Math.round(performance.now() - start) }, "TTS failed");
        return null;
    }
}

/**
 * TRUE streaming: send FULL text in ONE request, yield PCM chunks
 * as they arrive from the Deepgram stream. No sentence splitting.
 * Each chunk is immediately playable via AudioPublisher.pushPCM().
 *
 * Yields: { pcm: Int16Array, index: chunkNumber, text: fullText (first chunk only) }
 */
async function* deepgramGeneratePCMPipelined(text, sampleRate = "16000") {
    if (!text?.trim()) return;
    const start = performance.now();
    const model = process.env.DEEPGRAM_TTS_MODEL || "aura-2-en";

    try {
        ttsLog.info({ provider: "deepgram", textLen: text.length, model, streaming: true }, "TTS stream start");
        const dg = getDeepgramTTSClient();
        const response = await dg.speak.request(
            { text },
            {
                model,
                encoding: "linear16",
                sample_rate: parseInt(sampleRate),
                container: "none",
            }
        );
        const stream = await response.getStream();
        if (!stream) {
            ttsLog.error({ provider: "deepgram" }, "TTS returned no stream");
            return;
        }

        const reader = stream.getReader();
        let chunkIndex = 0;
        let totalSamples = 0;
        let leftover = Buffer.alloc(0); // buffer for incomplete samples (need even byte count)

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!value || value.length === 0) continue;

            // Combine with leftover bytes from previous chunk
            let buf = leftover.length > 0
                ? Buffer.concat([leftover, Buffer.from(value)])
                : Buffer.from(value);

            // Int16 = 2 bytes per sample — ensure even byte count
            const usableBytes = buf.length - (buf.length % 2);
            if (usableBytes === 0) {
                leftover = buf;
                continue;
            }

            leftover = buf.length > usableBytes ? buf.subarray(usableBytes) : Buffer.alloc(0);
            const pcm = new Int16Array(buf.buffer, buf.byteOffset, usableBytes / 2);

            if (pcm.length > 0) {
                totalSamples += pcm.length;
                // First chunk carries the text for socket.io emission
                yield { pcm, index: chunkIndex, text: chunkIndex === 0 ? text : "" };
                if (chunkIndex === 0) {
                    ttsLog.info({ provider: "deepgram", firstChunkMs: Math.round(performance.now() - start), samples: pcm.length }, "TTS first chunk");
                }
                chunkIndex++;
            }
        }

        // Flush any remaining leftover (unlikely but safe)
        if (leftover.length >= 2) {
            const usable = leftover.length - (leftover.length % 2);
            const pcm = new Int16Array(leftover.buffer, leftover.byteOffset, usable / 2);
            if (pcm.length > 0) {
                totalSamples += pcm.length;
                yield { pcm, index: chunkIndex, text: "" };
            }
        }

        ttsLog.info({ provider: "deepgram", totalChunks: chunkIndex, totalSamples, totalMs: Math.round(performance.now() - start) }, "TTS stream complete");
    } catch (err) {
        ttsLog.error({ provider: "deepgram", err: err.message, ms: Math.round(performance.now() - start) }, "TTS stream failed");
    }
}

// ═══════════════════════════════════════════════════════════════════
// PROVIDER: ELEVENLABS  (TRUE streaming — chunks yielded as they arrive)
// ═══════════════════════════════════════════════════════════════════

/** Single-shot: send full text, collect all audio, return Int16Array. */
async function elevenlabsGeneratePCM(text, sampleRate = "16000") {
    if (!text?.trim()) return null;
    const start = performance.now();
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) { ttsLog.error({ provider: "elevenlabs" }, "ELEVENLABS_API_KEY missing"); return null; }

    const voiceId = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";
    const modelId = process.env.ELEVENLABS_MODEL_ID || "eleven_turbo_v2_5";

    try {
        ttsLog.info({ provider: "elevenlabs", textLen: text.length, voice: voiceId, model: modelId }, "TTS request");
        const response = await fetch(
            `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json", "xi-api-key": apiKey },
                body: JSON.stringify({ text, model_id: modelId, output_format: `pcm_${sampleRate}` }),
            }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        const buffer = Buffer.from(await response.arrayBuffer());
        const pcm = bufferToInt16(buffer);
        ttsLog.info({ provider: "elevenlabs", samples: pcm.length, ms: Math.round(performance.now() - start) }, "TTS ready");
        return pcm;
    } catch (err) {
        ttsLog.error({ provider: "elevenlabs", err: err.message, ms: Math.round(performance.now() - start) }, "TTS failed");
        return null;
    }
}

/**
 * TRUE streaming: send FULL text in ONE request, yield PCM chunks
 * as they arrive from the ElevenLabs stream.
 */
async function* elevenlabsGeneratePCMPipelined(text, sampleRate = "16000") {
    if (!text?.trim()) return;
    const start = performance.now();
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) { ttsLog.error({ provider: "elevenlabs" }, "ELEVENLABS_API_KEY missing"); return; }

    const voiceId = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";
    const modelId = process.env.ELEVENLABS_MODEL_ID || "eleven_turbo_v2_5";

    try {
        ttsLog.info({ provider: "elevenlabs", textLen: text.length, voice: voiceId, streaming: true }, "TTS stream start");
        const response = await fetch(
            `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json", "xi-api-key": apiKey },
                body: JSON.stringify({ text, model_id: modelId, output_format: `pcm_${sampleRate}` }),
            }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);

        const reader = response.body.getReader();
        let chunkIndex = 0;
        let totalSamples = 0;
        let leftover = Buffer.alloc(0);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!value || value.length === 0) continue;

            let buf = leftover.length > 0
                ? Buffer.concat([leftover, Buffer.from(value)])
                : Buffer.from(value);

            const usableBytes = buf.length - (buf.length % 2);
            if (usableBytes === 0) { leftover = buf; continue; }

            leftover = buf.length > usableBytes ? buf.subarray(usableBytes) : Buffer.alloc(0);
            const pcm = new Int16Array(buf.buffer, buf.byteOffset, usableBytes / 2);

            if (pcm.length > 0) {
                totalSamples += pcm.length;
                yield { pcm, index: chunkIndex, text: chunkIndex === 0 ? text : "" };
                if (chunkIndex === 0) {
                    ttsLog.info({ provider: "elevenlabs", firstChunkMs: Math.round(performance.now() - start), samples: pcm.length }, "TTS first chunk");
                }
                chunkIndex++;
            }
        }

        if (leftover.length >= 2) {
            const usable = leftover.length - (leftover.length % 2);
            const pcm = new Int16Array(leftover.buffer, leftover.byteOffset, usable / 2);
            if (pcm.length > 0) { totalSamples += pcm.length; yield { pcm, index: chunkIndex, text: "" }; }
        }

        ttsLog.info({ provider: "elevenlabs", totalChunks: chunkIndex, totalSamples, totalMs: Math.round(performance.now() - start) }, "TTS stream complete");
    } catch (err) {
        ttsLog.error({ provider: "elevenlabs", err: err.message, ms: Math.round(performance.now() - start) }, "TTS stream failed");
    }
}

// ═══════════════════════════════════════════════════════════════════
// PROVIDER REGISTRY — switch via INTERVIEW_TTS_PROVIDER env var
// ═══════════════════════════════════════════════════════════════════

const providers = {
    polly: {
        generatePCM: pollyGeneratePCM,
        generatePCMPipelined: pollyGeneratePCMPipelined,
    },
    deepgram: {
        generatePCM: deepgramGeneratePCM,
        generatePCMPipelined: deepgramGeneratePCMPipelined,
    },
    elevenlabs: {
        generatePCM: elevenlabsGeneratePCM,
        generatePCMPipelined: elevenlabsGeneratePCMPipelined,
    },
};

function getProvider() {
    const name = (process.env.INTERVIEW_TTS_PROVIDER || "polly").toLowerCase();
    const provider = providers[name];
    if (!provider) {
        ttsLog.warn({ requested: name, available: Object.keys(providers) }, "Unknown TTS provider, falling back to polly");
        return providers.polly;
    }
    return provider;
}

// Log which provider is active at import time
const activeProviderName = (process.env.INTERVIEW_TTS_PROVIDER || "polly").toLowerCase();
ttsLog.info({ provider: activeProviderName }, "Interview TTS provider loaded");

// ═══════════════════════════════════════════════════════════════════
// PUBLIC API — same signatures as before, worker.js doesn't change
// ═══════════════════════════════════════════════════════════════════

export async function generatePCM(text, sampleRate = "16000") {
    return getProvider().generatePCM(text, sampleRate);
}

export async function* generatePCMPipelined(text, sampleRate = "16000") {
    yield* getProvider().generatePCMPipelined(text, sampleRate);
}

/**
 * ElevenLabs: Get word-level timestamps via the /with-timestamps endpoint.
 * Returns alignment data with character-level timing, mapped to word boundaries.
 */
async function elevenlabsGetSpeechMarks(text) {
    if (!text?.trim()) return [];
    const apiKey = process.env.ELEVENLABS_API_KEY;
    if (!apiKey) return estimateWordTimings(text);

    const voiceId = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";
    const modelId = process.env.ELEVENLABS_MODEL_ID || "eleven_turbo_v2_5";

    try {
        const response = await fetch(
            `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/with-timestamps`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json", "xi-api-key": apiKey },
                body: JSON.stringify({
                    text,
                    model_id: modelId,
                    output_format: "pcm_16000",
                }),
            }
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        // ElevenLabs returns NDJSON with alignment + audio_base64 chunks
        const body = await response.text();
        const lines = body.split("\n").filter(l => l.trim());
        const marks = [];

        for (const line of lines) {
            try {
                const chunk = JSON.parse(line);
                if (chunk.alignment) {
                    const { characters, character_start_times_seconds, character_end_times_seconds } = chunk.alignment;
                    if (!characters || !character_start_times_seconds) continue;

                    // Reconstruct words from character-level data
                    let wordStart = 0;
                    let currentWord = "";
                    for (let i = 0; i < characters.length; i++) {
                        const ch = characters[i];
                        if (ch === " " || i === characters.length - 1) {
                            if (i === characters.length - 1 && ch !== " ") currentWord += ch;
                            if (currentWord.trim()) {
                                marks.push({
                                    time: Math.round(character_start_times_seconds[wordStart] * 1000),
                                    value: currentWord.trim(),
                                });
                            }
                            currentWord = "";
                            wordStart = i + 1;
                        } else {
                            currentWord += ch;
                        }
                    }
                }
            } catch { /* skip unparseable lines */ }
        }

        if (marks.length > 0) {
            ttsLog.info({ provider: "elevenlabs", words: marks.length }, "SpeechMarks from alignment");
            return marks;
        }
        return estimateWordTimings(text);
    } catch (err) {
        ttsLog.warn({ provider: "elevenlabs", err: err.message }, "Alignment failed, using estimation");
        return estimateWordTimings(text);
    }
}

/**
 * Deepgram: Compute word timestamps from PCM audio duration.
 * Since Deepgram REST doesn't return word timing, we use the actual audio
 * sample count to calculate total duration, then distribute words evenly.
 * This is more accurate than blind estimation because we know the real duration.
 */
async function deepgramGetSpeechMarks(text) {
    if (!text?.trim()) return [];

    // Generate a quick PCM to get actual audio duration
    try {
        const pcm = await deepgramGeneratePCM(text, "16000");
        if (pcm) {
            const durationMs = (pcm.length / 16000) * 1000;
            ttsLog.info({ provider: "deepgram", durationMs: Math.round(durationMs) }, "SpeechMarks from audio duration");
            return estimateWordTimings(text, durationMs);
        }
    } catch { /* fall through */ }

    return estimateWordTimings(text);
}

/**
 * Get word-level timestamps for subtitle sync.
 * - Polly:      real SpeechMarks API (most accurate, parallel request)
 * - Deepgram:   duration-based estimation (uses actual audio length)
 * - ElevenLabs: alignment endpoint (real timestamps, slight delay ~100-200ms)
 *
 * Returns: [{ time: ms, value: "word" }, ...]
 */
export async function getSpeechMarks(text, audioDurationMs = null) {
    const provider = (process.env.INTERVIEW_TTS_PROVIDER || "polly").toLowerCase();

    if (provider === "polly") {
        return pollyGetSpeechMarks(text);
    }
    if (provider === "elevenlabs") {
        return elevenlabsGetSpeechMarks(text);
    }
    if (provider === "deepgram") {
        // If caller already knows the duration (from streamed PCM), use it
        if (audioDurationMs) return estimateWordTimings(text, audioDurationMs);
        return deepgramGetSpeechMarks(text);
    }
    return estimateWordTimings(text, audioDurationMs);
}
