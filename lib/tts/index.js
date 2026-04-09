/**
 * TTS Provider Registry
 *
 * Reads TTS_PROVIDER from .env and loads the matching provider ONCE at startup.
 * All app code imports textToAudio() from here — never from providers directly.
 *
 * Supported providers:
 *   TTS_PROVIDER=polly    → Amazon Polly (default, cloud API, no Docker needed)
 *   TTS_PROVIDER=kokoro   → Kokoro-FastAPI (self-hosted, requires Docker)
 *
 * Adding a new provider (e.g. ElevenLabs):
 *   1. Create lib/tts/providers/elevenlabsProvider.js  (export textToAudio)
 *   2. Add "elevenlabs" entry to providerMap below
 *   3. Set TTS_PROVIDER=elevenlabs in .env
 *   → Done. Zero changes anywhere else.
 */

import dotenv from 'dotenv';
import { ttsLog } from '../logger.js';
dotenv.config();

const PROVIDER = (process.env.TTS_PROVIDER || "polly").toLowerCase();

const providerMap = {
    polly: () => import("./providers/pollyProvider.js"),
    kokoro: () => import("./providers/kokoroProvider.js"),
    gemini: () => import("./providers/geminiProvider.js"),
    // elevenlabs: () => import("./providers/elevenlabsProvider.js"),  ← future
};

if (!providerMap[PROVIDER]) {
    throw new Error(
        `[TTS] Unknown TTS_PROVIDER: "${PROVIDER}". ` +
        `Valid options: ${Object.keys(providerMap).join(", ")}`
    );
}

ttsLog.info({ provider: PROVIDER }, 'Loading TTS provider');

// Load the provider module ONCE at startup (not per-request)
const providerModule = await providerMap[PROVIDER]();

ttsLog.info({ provider: PROVIDER }, 'TTS provider ready');

/**
 * Convert text to a base64-encoded audio string.
 * @param {string} text
 * @returns {Promise<string>} base64 audio
 */
export const textToAudio = providerModule.textToAudio;

/**
 * MIME type of the audio returned by textToAudio().
 * Used by API routes to build correct data URIs.
 * Defaults to audio/mp3 for providers that don't export it.
 */
export const audioMimeType = providerModule.audioMimeType || 'audio/mp3';
