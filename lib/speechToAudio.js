/**
 * Thin re-export — all existing imports of this file keep working unchanged.
 * The actual implementation lives in lib/tts/index.js (provider registry).
 * To switch TTS engine, just change TTS_PROVIDER in your .env file.
 */
export { textToAudio } from "./tts/index.js";
