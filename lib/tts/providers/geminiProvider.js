import { GoogleGenAI, Modality } from '@google/genai';
import { Mp3Encoder } from '@breezystack/lamejs';
import dotenv from 'dotenv';
import { ttsLog } from '../../logger.js';

dotenv.config();

/**
 * Gemini Live TTS Provider
 *
 * Uses the same Gemini Live native-audio model as the voice agent,
 * but for one-shot text-to-speech synthesis (used for voice summary).
 *
 * Voice options (female): Kore, Aoede, Leda, Zephyr
 * Voice options (male): Puck, Charon, Fenrir, Orus
 *
 * Gemini Live outputs 24kHz PCM mono. We wrap it in a WAV header so
 * browsers can play it natively via <audio> tag.
 *
 * Env vars:
 *   GEMINI_API_KEY         — required
 *   GEMINI_LIVE_MODEL      — default: gemini-2.5-flash-native-audio-latest
 *   GEMINI_TTS_VOICE       — default: Aoede (female)
 */

const VOICE_MODEL = process.env.GEMINI_LIVE_MODEL || 'gemini-2.5-flash-native-audio-latest';
const VOICE_NAME = process.env.GEMINI_TTS_VOICE || 'Aoede';
const OUTPUT_SAMPLE_RATE = 24000;

/**
 * Encode raw PCM16 mono audio buffer into MP3 using lamejs.
 * @param {Buffer} pcmBuffer - raw PCM16 little-endian samples
 * @param {number} sampleRate - source sample rate (Hz)
 * @returns {Buffer} MP3 audio
 */
function pcmToMp3(pcmBuffer, sampleRate = OUTPUT_SAMPLE_RATE) {
    const numChannels = 1;
    const kbps = 128;
    const encoder = new Mp3Encoder(numChannels, sampleRate, kbps);

    // Convert Buffer → Int16Array (little-endian, no copy)
    const samples = new Int16Array(
        pcmBuffer.buffer,
        pcmBuffer.byteOffset,
        pcmBuffer.byteLength / 2
    );

    const sampleBlockSize = 1152; // Required by MP3 frame layout
    const mp3Chunks = [];

    for (let i = 0; i < samples.length; i += sampleBlockSize) {
        const chunk = samples.subarray(i, i + sampleBlockSize);
        const mp3buf = encoder.encodeBuffer(chunk);
        if (mp3buf.length > 0) mp3Chunks.push(Buffer.from(mp3buf));
    }
    const mp3End = encoder.flush();
    if (mp3End.length > 0) mp3Chunks.push(Buffer.from(mp3End));

    return Buffer.concat(mp3Chunks);
}

/**
 * Generate one-shot audio from text using Gemini Live API.
 */
export async function textToAudio(text) {
    if (!text || !text.trim()) {
        throw new Error('[Gemini TTS] textToAudio requires non-empty text');
    }

    const start = performance.now();
    ttsLog.info({ provider: 'gemini', voice: VOICE_NAME, textLength: text.length }, 'Gemini TTS synthesis started');

    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    const audioChunks = [];
    let resolveAudio, rejectAudio;
    const audioPromise = new Promise((res, rej) => {
        resolveAudio = res;
        rejectAudio = rej;
    });

    // Safety timeout
    const timeout = setTimeout(() => {
        rejectAudio(new Error('[Gemini TTS] Timeout after 60s waiting for audio'));
    }, 60_000);

    let session;
    try {
        session = await ai.live.connect({
            model: VOICE_MODEL,
            config: {
                responseModalities: [Modality.AUDIO],
                systemInstruction: {
                    parts: [{
                        text: 'You are a narration engine. Read the user\'s text aloud verbatim, exactly as written. Do not add greetings, commentary, summaries, or any extra words. Just speak the text.'
                    }]
                },
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: VOICE_NAME }
                    }
                }
            },
            callbacks: {
                onopen: () => {
                    ttsLog.debug({ provider: 'gemini' }, 'Gemini Live TTS connection opened');
                },
                onmessage: (msg) => {
                    if (msg.serverContent?.modelTurn?.parts) {
                        for (const part of msg.serverContent.modelTurn.parts) {
                            if (part.inlineData?.mimeType?.startsWith('audio/')) {
                                audioChunks.push(Buffer.from(part.inlineData.data, 'base64'));
                            }
                        }
                    }
                    if (msg.serverContent?.turnComplete || msg.serverContent?.generationComplete) {
                        clearTimeout(timeout);
                        if (audioChunks.length === 0) {
                            rejectAudio(new Error('[Gemini TTS] No audio received'));
                            return;
                        }
                        const pcm = Buffer.concat(audioChunks);
                        const encodeStart = performance.now();
                        const mp3 = pcmToMp3(pcm);
                        const encodeMs = Math.round(performance.now() - encodeStart);
                        const durationMs = Math.round(performance.now() - start);
                        ttsLog.info({
                            provider: 'gemini',
                            voice: VOICE_NAME,
                            durationMs,
                            encodeMs,
                            mp3Bytes: mp3.length,
                            pcmBytes: pcm.length,
                        }, 'Gemini TTS complete');
                        resolveAudio(mp3.toString('base64'));
                    }
                },
                onerror: (e) => {
                    clearTimeout(timeout);
                    rejectAudio(new Error(`[Gemini TTS] ${e?.message || String(e)}`));
                },
                onclose: () => {
                    if (audioChunks.length > 0) {
                        clearTimeout(timeout);
                        const pcm = Buffer.concat(audioChunks);
                        const mp3 = pcmToMp3(pcm);
                        resolveAudio(mp3.toString('base64'));
                    }
                }
            }
        });

        // Now session is defined — send the text
        session.sendClientContent({
            turns: [{ role: 'user', parts: [{ text }] }],
            turnComplete: true
        });

        const result = await audioPromise;
        try { session.close(); } catch {}
        return result;

    } catch (err) {
        clearTimeout(timeout);
        try { session?.close(); } catch {}
        throw err;
    }
}

/**
 * Returns the audio MIME type this provider produces.
 * Used by the summary route to set the correct data URI.
 */
export const audioMimeType = 'audio/mp3';
