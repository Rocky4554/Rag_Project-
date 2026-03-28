import {
    PollyClient,
    SynthesizeSpeechCommand,
} from "@aws-sdk/client-polly";
import dotenv from "dotenv";
import { ttsLog } from "../../lib/logger.js";
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

    const start = performance.now();
    const voiceId = process.env.POLLY_VOICE_ID || 'Joanna';
    const engine = process.env.POLLY_ENGINE || 'neural';
    const region = process.env.AWS_REGION || 'us-east-1';
    try {
        ttsLog.info({ textLength: text.length, text: text.substring(0, 50), voice: voiceId, engine, region }, 'TTS Polly request');

        const command = new SynthesizeSpeechCommand({
            Text: text,
            OutputFormat: "pcm",
            SampleRate: sampleRate,
            VoiceId: voiceId,
            Engine: engine,
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

        const durationMs = Math.round(performance.now() - start);
        ttsLog.info({ samples: int16Array.length, durationMs, voice: voiceId, engine }, 'TTS Polly ready');
        return int16Array;

    } catch (err) {
        const durationMs = Math.round(performance.now() - start);
        ttsLog.error(
            { durationMs, err: err.message, region: process.env.AWS_REGION, voice: process.env.POLLY_VOICE_ID, engine: process.env.POLLY_ENGINE },
            'Agent TTS PCM generation failed'
        );
        return null;
    }
}

/**
 * Split text into sentences for pipelined TTS.
 * Returns array of non-empty sentence strings.
 */
export function splitIntoSentences(text) {
    if (!text) return [];
    // Split on sentence-ending punctuation followed by space or end of string
    const sentences = text.match(/[^.!?]+[.!?]+[\s]?|[^.!?]+$/g);
    if (!sentences) return [text.trim()];
    return sentences.map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Generates PCM for multiple sentences in a pipelined fashion.
 * Yields { pcm, index } as each sentence's audio becomes ready.
 * The first sentence is prioritized — caller can start playback immediately.
 */
export async function* generatePCMPipelined(text, sampleRate = "16000") {
    const sentences = splitIntoSentences(text);
    if (sentences.length === 0) return;

    if (sentences.length === 1) {
        const pcm = await generatePCM(sentences[0], sampleRate);
        if (pcm) yield { pcm, index: 0, text: sentences[0] };
        return;
    }

    ttsLog.info({ sentences: sentences.length, totalChars: text.length }, 'TTS pipeline start');

    // Start ALL Polly requests concurrently
    const sentenceStartTs = performance.now();
    const promises = sentences.map((sentence, i) => {
        const sentTs = performance.now();
        ttsLog.info({ idx: i, sentence: sentence.substring(0, 60) }, 'TTS sentence dispatched');
        return generatePCM(sentence, sampleRate).then(pcm => {
            ttsLog.info({ idx: i, durationMs: Math.round(performance.now() - sentTs), voice: process.env.POLLY_VOICE_ID || 'Joanna' }, 'TTS sentence ready');
            return { pcm, index: i };
        });
    });

    // Yield in order as they complete, but prioritize sequential order
    // so audio plays in the right sequence
    const results = new Array(sentences.length);
    let nextToYield = 0;

    // Create a resolver for each index
    const resolvers = new Array(sentences.length);
    const readyPromises = sentences.map((_, i) =>
        new Promise(resolve => { resolvers[i] = resolve; })
    );

    // As each promise completes, mark it ready
    for (const p of promises) {
        p.then(result => {
            results[result.index] = result;
            resolvers[result.index]();
        });
    }

    // Yield in sequential order
    for (let i = 0; i < sentences.length; i++) {
        await readyPromises[i];
        if (results[i]?.pcm) {
            yield { ...results[i], text: sentences[i] };
        }
    }
}
