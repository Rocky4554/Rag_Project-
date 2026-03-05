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
        maxOutputTokens: 2048,
    });

    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are an expert summarizer. Your task is to extract the core ideas, main topics, and essential details from the following document context, and write a clear, concise, and easy-to-read summary. The summary should be written in plain text (no markdown formatting, no bullet points, no bold text) so it can be cleanly read aloud by a Text-to-Speech engine.

Context:
{context}

Requirements:
1. Provide a comprehensive but concise summary of the entire context.
2. Output ONLY the summary text.
3. DO NOT use markdown, asterisks, hash tags, or bullet points. Write in complete paragraphs.`
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

export async function textToAudio(text) {
    const apiUrl = process.env.KOKORO_API_URL || "http://localhost:8880";
    console.log(`Sending text to Kokoro TTS at ${apiUrl}...`);

    // Use smaller chunks (~400 chars). Kokoro CPU inference drastically slows down on very large inputs.
    // Breaking it down into paragraph-sized chunks allows it to return segments much faster.
    const MAX_CHARS = 400;
    const chunks = splitBySentence(text, MAX_CHARS);
    const audioBuffers = [];

    // Process chunks. We do this sequentially right now to avoid OOM or CPU locking the Docker container
    console.log(`Split summary into ${chunks.length} chunks for TTS.`);
    console.time("Total TTS Generation");

    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        if (!chunk.trim()) continue;

        console.log(`\nProcessing TTS chunk ${i + 1} of ${chunks.length} (${chunk.length} chars)...`);

        let arrayBuffer = null;
        let retries = 3;

        console.time(`Chunk ${i + 1} processing`);
        while (retries > 0) {
            try {
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

                arrayBuffer = await response.arrayBuffer();
                break; // Success, exit retry loop
            } catch (err) {
                console.error(`Chunk ${i + 1} failed: ${err.message}`);
                retries--;
                if (retries === 0) {
                    throw new Error(`Failed to generate TTS after 3 retries: ${err.message}`);
                }
                console.log(`Retrying chunk ${i + 1}... (${retries} retries left). Waiting 2 seconds...`);
                await new Promise(res => setTimeout(res, 2000));
            }
        }
        console.timeEnd(`Chunk ${i + 1} processing`);

        audioBuffers.push(Buffer.from(arrayBuffer));
    }
    console.timeEnd("Total TTS Generation");

    console.log("\nMerging audio buffers...");
    console.time("Audio Merging");
    const combinedBuffer = Buffer.concat(audioBuffers);
    console.timeEnd("Audio Merging");

    console.log("Merging completed. Ready to play!");

    return combinedBuffer.toString("base64");
}
