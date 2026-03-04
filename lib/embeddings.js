import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import dotenv from 'dotenv';
dotenv.config();

if (!process.env.GEMINI_API_KEY) {
    console.warn("Missing GEMINI_API_KEY in environment variables");
}

console.log("API Key loaded:", !!process.env.GEMINI_API_KEY); // should print true

// LangChain provides a built-in wrapper that automatically
// handles batching and rate limits for us!
export const embedder = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "gemini-embedding-001",
});