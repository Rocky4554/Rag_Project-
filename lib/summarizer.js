import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';
dotenv.config();

export async function summarizeDocs(docs) {
    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    console.log(`Setting up LangChain summarization for ${docs.length} document chunks...`);

    const llm = new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: "gemini-2.5-flash",
        temperature: 0.3,
        maxOutputTokens: 8192, // Increased from 2048 to prevent large summaries from being cut off mid-sentence
    });

    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are an expert summarizer. Read the following document and write a short, concise summary in plain text suitable for text-to-speech.

Context:
{context}

Requirements:
1. Keep the summary SHORT — under 300 words. Cover only the most important points.
2. Output ONLY the summary text.
3. DO NOT use markdown, asterisks, bullet points, or headers. Write in plain paragraphs.`
        ],
        ["human", "Summarize the text."]
    ]);

    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: prompt,
    });

    console.log("Generating summary with Gemini...");
    console.time("Gemini Summary Generation");
    const result = await combineDocsChain.invoke({
        context: docs
    });
    console.timeEnd("Gemini Summary Generation");

    console.log("Summary generated successfully!");
    return result.trim();
}

function splitBySentence(text, max) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];
    let cur = "";
    for (const s of sentences) {
        // Kokoro-FastAPI max length is typically around 400-500 chars per request for optimal CPU generation.
        // We will target ~400 chars per chunk to keep the model fast and prevent it from timing out or hanging
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

// Helper: fetch one chunk with retry logic
async function fetchChunk(apiUrl, chunk, index) {
    let retries = 3;
    while (retries > 0) {
        try {
            console.log(`Processing TTS chunk ${index + 1} (${chunk.length} chars)...`);
            const response = await fetch(`${apiUrl}/v1/audio/speech`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: "kokoro",
                    input: chunk,
                    voice: "af_heart",
                    response_format: "mp3"
                })
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`TTS API Error (${response.status}): ${errText}`);
            }

            const arrayBuffer = await response.arrayBuffer();
            console.log(`Chunk ${index + 1} done.`);
            return Buffer.from(arrayBuffer);
        } catch (err) {
            console.error(`Chunk ${index + 1} failed: ${err.message}`);
            retries--;
            if (retries === 0) throw new Error(`Chunk ${index + 1} failed after 3 retries: ${err.message}`);
            console.log(`Retrying chunk ${index + 1}... (${retries} left). Waiting 2s...`);
            await new Promise(res => setTimeout(res, 2000));
        }
    }
}

// Run up to `limit` Promises concurrently, preserving output order
async function parallelLimit(tasks, limit) {
    const results = new Array(tasks.length);
    let index = 0;

    async function worker() {
        while (index < tasks.length) {
            const i = index++;
            results[i] = await tasks[i]();
        }
    }

    // Spin up `limit` workers that race through the task queue
    await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
    return results;
}

export async function textToAudio(text) {
    const apiUrl = process.env.KOKORO_API_URL || "http://localhost:8880";
    // CPU Kokoro uses ALL cores for one request — parallel requests fight over the same
    // CPU threads and end up slower, not faster. Keep CONCURRENCY=1 (sequential).
    // If you switch to GPU Kokoro, bump this to 3-5 for a real speedup.
    const MAX_CHARS = 700;  // Larger chunks = fewer HTTP round-trips to Kokoro
    const CONCURRENCY = 1; // Sequential is optimal for single-CPU Docker container

    const chunks = splitBySentence(text, MAX_CHARS).filter(c => c.trim());
    console.log(`Split summary into ${chunks.length} chunks (sequential, CPU mode)...`);
    console.time("Total TTS Generation");

    const tasks = chunks.map((chunk, i) => () => fetchChunk(apiUrl, chunk, i));
    const audioBuffers = await parallelLimit(tasks, CONCURRENCY);

    console.timeEnd("Total TTS Generation");

    console.log("Merging audio buffers...");
    const combinedBuffer = Buffer.concat(audioBuffers);
    console.log("Done. Ready to play!");

    return combinedBuffer.toString("base64");
}
