import { spawn } from 'child_process';
import fs from 'fs';
import { agentLog } from '../../lib/logger.js';

// Absolute fallback path for the winget-installed ffmpeg.
// Winget adds it to the user PATH, but that update only takes effect in new shell
// sessions — this ensures the first server start after install still works.
const FFMPEG_WINGET = 'C:\\Users\\Deedar\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe';

function resolveFFmpeg() {
    // Prefer PATH; fall back to known winget location if it exists.
    if (fs.existsSync(FFMPEG_WINGET)) return FFMPEG_WINGET;
    return 'ffmpeg';
}

/**
 * Decodes an MP3 file to raw 16-bit mono PCM (Int16Array) using ffmpeg.
 * Returns null if ffmpeg is unavailable or decoding fails — caller falls
 * back to Polly TTS in that case.
 *
 * @param {string} filePath   - Absolute path to the .mp3 file
 * @param {number} sampleRate - Target sample rate (default 16000 to match AudioPublisher)
 * @returns {Promise<Int16Array|null>}
 */
export async function mp3ToPCM(filePath, sampleRate = 16000) {
    return new Promise((resolve) => {
        const chunks = [];
        const cmd = resolveFFmpeg();

        const proc = spawn(cmd, [
            '-i', filePath,
            '-f', 's16le',         // signed 16-bit little-endian PCM
            '-ar', String(sampleRate),
            '-ac', '1',            // mono
            'pipe:1'               // write raw PCM to stdout
        ], { stdio: ['ignore', 'pipe', 'pipe'] });

        proc.stdout.on('data', (chunk) => chunks.push(Buffer.from(chunk)));

        proc.on('close', (code) => {
            if (code !== 0 || chunks.length === 0) {
                agentLog.warn({ filePath, code }, 'mp3ToPCM: ffmpeg decode failed');
                return resolve(null);
            }
            const buf = Buffer.concat(chunks);
            const pcm = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
            agentLog.info({ filePath, samples: pcm.length, sampleRate }, 'mp3ToPCM: decoded OK');
            resolve(pcm);
        });

        proc.on('error', (err) => {
            agentLog.warn({ err: err.message, filePath }, 'mp3ToPCM: ffmpeg spawn failed');
            resolve(null);
        });
    });
}
