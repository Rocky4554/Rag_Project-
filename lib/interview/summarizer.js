import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { detectRateLimitError } from "../llm.js";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { summaryLog } from "../logger.js";
import dotenv from 'dotenv';
import { textToAudio } from '../tts/speechToAudio.js';
dotenv.config();

const SUMMARIZER_MODEL = "gemini-2.5-flash";

export async function summarizeDocs(docs) {
    const start = performance.now();

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    const totalInputChars = docs.reduce((sum, d) => sum + (d.pageContent?.length || 0), 0);
    summaryLog.info({ docChunks: docs.length, totalInputChars }, 'Summarization started');

    const llm = new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: SUMMARIZER_MODEL,
        temperature: 0.3,
        maxOutputTokens: 8192,
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

    summaryLog.info({ model: SUMMARIZER_MODEL }, 'Summary LLM call started');
    const llmStart = performance.now();

    let result;
    try {
        result = await combineDocsChain.invoke({
            context: docs
        });
    } catch (err) {
        const llmMs = Math.round(performance.now() - llmStart);
        const rateLimitType = detectRateLimitError(err);
        if (rateLimitType) {
            summaryLog.error({ rateLimitType, provider: 'google', llmMs, err: err.message }, 'Summary LLM rate limit / quota error');
        } else {
            summaryLog.error({ llmMs, err: err.message }, 'Summary LLM call failed');
        }
        throw err;
    }

    const llmMs = Math.round(performance.now() - llmStart);
    const summary = result.trim();
    const totalMs = Math.round(performance.now() - start);

    summaryLog.info(
        { totalMs, llmMs, inputChunks: docs.length, outputChars: summary.length, outputWords: summary.split(/\s+/).length },
        'Summarization complete'
    );

    return summary;
}

export { textToAudio };
