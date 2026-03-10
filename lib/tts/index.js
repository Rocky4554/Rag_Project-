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
dotenv.config();

const PROVIDER = (process.env.TTS_PROVIDER || "polly").toLowerCase();

const providerMap = {
    polly: () => import("./providers/pollyProvider.js"),
    kokoro: () => import("./providers/kokoroProvider.js"),
    // elevenlabs: () => import("./providers/elevenlabsProvider.js"),  ← future
};

if (!providerMap[PROVIDER]) {
    throw new Error(
        `[TTS] Unknown TTS_PROVIDER: "${PROVIDER}". ` +
        `Valid options: ${Object.keys(providerMap).join(", ")}`
    );
}

console.log(`[TTS] Loading provider: "${PROVIDER}"...`);

// Load the provider module ONCE at startup (not per-request)
const providerModule = await providerMap[PROVIDER]();

console.log(`[TTS] Provider "${PROVIDER}" ready.`);

/**
 * Convert text to a base64-encoded MP3 string.
 * @param {string} text
 * @returns {Promise<string>} base64 MP3
 */
export const textToAudio = providerModule.textToAudio;
