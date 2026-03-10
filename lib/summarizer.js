import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';
import { textToAudio } from './speechToAudio.js';
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

export { textToAudio };
