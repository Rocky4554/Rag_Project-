import { GoogleGenAI, Modality } from '@google/genai';
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
 * Wrap raw PCM16 mono audio in a WAV container so browsers can play it.
 */
function pcmToWav(pcmBuffer, sampleRate = OUTPUT_SAMPLE_RATE) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataSize = pcmBuffer.length;

    const header = Buffer.alloc(44);
    // RIFF chunk
    header.write('RIFF', 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write('WAVE', 8);
    // fmt chunk
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);           // fmt chunk size
    header.writeUInt16LE(1, 20);            // PCM format
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    // data chunk
    header.write('data', 36);
    header.writeUInt32LE(dataSize, 40);

    return Buffer.concat([header, pcmBuffer]);
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

    return new Promise((resolve, reject) => {
        const audioChunks = [];
        let session = null;
        let settled = false;

        const finish = (err, result) => {
            if (settled) return;
            settled = true;
            try { session?.close(); } catch {}
            if (err) reject(err);
            else resolve(result);
        };

        // Safety timeout — don't hang if Gemini never responds
        const timeout = setTimeout(() => {
            finish(new Error('[Gemini TTS] Timeout after 60s waiting for audio'));
        }, 60_000);

        ai.live.connect({
            model: VOICE_MODEL,
            config: {
                responseModalities: [Modality.AUDIO],
                thinkingConfig: { thinkingBudget: 0 },
                systemInstruction: {
                    parts: [{
                        text: 'You are a text-to-speech narrator. Read the following text naturally and clearly, exactly as written. Do not add any commentary, greetings, or extra words. Just narrate the text.'
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
                    // Send the text to be narrated
                    session.sendClientContent({
                        turns: [{ role: 'user', parts: [{ text }] }],
                        turnComplete: true
                    });
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
                            finish(new Error('[Gemini TTS] No audio received'));
                            return;
                        }
                        const pcm = Buffer.concat(audioChunks);
                        const wav = pcmToWav(pcm);
                        const durationMs = Math.round(performance.now() - start);
                        ttsLog.info({
                            provider: 'gemini',
                            voice: VOICE_NAME,
                            durationMs,
                            audioBytes: wav.length,
                            pcmBytes: pcm.length,
                        }, 'Gemini TTS complete');
                        finish(null, wav.toString('base64'));
                    }
                },
                onerror: (e) => {
                    clearTimeout(timeout);
                    finish(new Error(`[Gemini TTS] ${e?.message || String(e)}`));
                },
                onclose: () => {
                    if (!settled && audioChunks.length > 0) {
                        clearTimeout(timeout);
                        const pcm = Buffer.concat(audioChunks);
                        const wav = pcmToWav(pcm);
                        finish(null, wav.toString('base64'));
                    } else if (!settled) {
                        clearTimeout(timeout);
                        finish(new Error('[Gemini TTS] Connection closed without audio'));
                    }
                }
            }
        }).then((s) => { session = s; }).catch((err) => {
            clearTimeout(timeout);
            finish(new Error(`[Gemini TTS] Failed to connect: ${err.message}`));
        });
    });
}

/**
 * Returns the audio MIME type this provider produces.
 * Used by the summary route to set the correct data URI.
 */
export const audioMimeType = 'audio/wav';
