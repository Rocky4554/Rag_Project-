/**
 * PRE-GENERATE TTS AUDIO CACHE
 * 
 * Run this ONCE. It calls your Kokoro TTS API for each phrase and saves .mp3 files.
 * After this, your interview agent plays the files directly — zero TTS cost for fillers.
 * 
 * Run: node generateTTSCache.js
 */

import fs from "fs";
import path from "path";
import dotenv from "dotenv";
dotenv.config();

// ── The phrases to pre-generate ───────────────────────────────────────────────
// key = what your code references, text = what gets spoken
const PHRASES = [
    { key: "lets_move_on", text: "Let's move on." },
    { key: "great_answer", text: "Great answer." },
    { key: "good_effort", text: "Good effort." },
    { key: "next_question", text: "Next question." },
    { key: "interesting", text: "Interesting." },

    // Edge case intents — full standalone responses (no LLM needed after these)
    { key: "take_your_time", text: "Take your time, there is absolutely no rush. Interviews can be stressful and it is completely normal to feel a bit nervous. Just take a deep breath, and whenever you are ready, I am here to listen." },
    { key: "no_worries", text: "No worries at all. Your health and well-being come first. We will end the interview right here, and I genuinely hope you feel better soon. Take care of yourself." },
    { key: "thats_okay", text: "That is completely okay. Not every question needs a perfect answer, and it takes honesty to admit when you are unsure. Let us simply move on to the next question." },
    { key: "out_of_context", text: "I am sorry, that seems to be a bit outside the scope of our interview today. Let us try to stay focused on the technical questions so we can make the most of our time together. Shall we carry on?" },
    { key: "interview_stopped", text: "Of course. As you have requested, we will go ahead and end the interview here. Thank you so much for your time today, it was truly a pleasure speaking with you. I wish you all the very best." },

    // Full intro / outro — played once per interview
    { key: "interview_intro", text: "Hello! Welcome to your AI voice interview. I will ask you 5 questions based on the document you uploaded. After I finish asking each question, your microphone will activate automatically. Please answer clearly. Let's begin!" },
    { key: "interview_outro", text: "There, I want to thank you for taking the time to speak with me today, I appreciate the opportunity to have learned more about your background and experiences. It was great getting to know you, and I wish you all the best in your future endeavors." },
];

const OUTPUT_DIR = "./assets/tts";

// Make sure the output folder exists
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// ═════════════════════════════════════════════════════════════════
// OPTION - Kokoro TTS (running locally)
// ═════════════════════════════════════════════════════════════════

const MAX_CHARS = 700;
const MAX_RETRIES = 3;

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
                    voice: process.env.KOKORO_VOICE || "af_heart", // Defaults to af_heart
                    response_format: "mp3"
                })
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`TTS API Error (${response.status}): ${errText}`);
            }

            const arrayBuffer = await response.arrayBuffer();
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

async function generateWithKokoro(text) {
    const apiUrl = process.env.KOKORO_API_URL || "http://localhost:8880";
    const chunks = splitBySentence(text, MAX_CHARS).filter(c => c.trim());

    const audioBuffers = [];
    for (let i = 0; i < chunks.length; i++) {
        audioBuffers.push(await fetchChunk(apiUrl, chunks[i], i));
    }

    return Buffer.concat(audioBuffers);
}


// ═════════════════════════════════════════════════════════════════
// MAIN — Runs all phrases through Kokoro
// ═════════════════════════════════════════════════════════════════

async function main() {
    console.log(`\n🎙️  Pre-generating ${PHRASES.length} TTS phrases with Kokoro (locally)...\n`);

    let success = 0;
    let skipped = 0;

    for (const { key, text } of PHRASES) {
        const outputPath = path.join(OUTPUT_DIR, `${key}.mp3`);

        // Skip if already generated — safe to re-run the script
        if (fs.existsSync(outputPath)) {
            console.log(`⏭  Skipped (already exists): ${key}.mp3`);
            skipped++;
            continue;
        }

        try {
            process.stdout.write(`⏳ Generating: "${text}" → ${key}.mp3 ... `);
            const audioBuffer = await generateWithKokoro(text);
            fs.writeFileSync(outputPath, audioBuffer);
            console.log(`✅`);
            success++;

            // Small delay between API calls
            await new Promise(r => setTimeout(r, 500));

        } catch (err) {
            console.log(`❌ FAILED: ${err.message}`);
        }
    }

    console.log(`\n✅ Done! ${success} generated, ${skipped} skipped.`);
    console.log(`📁 Files saved to: ${path.resolve(OUTPUT_DIR)}\n`);

    if (success + skipped === PHRASES.length) {
        console.log("🎉 All phrases ready. Your interview agent will now use cached audio.");
    }
}

main().catch(console.error);
