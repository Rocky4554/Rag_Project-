// test-embed.js
import dotenv from 'dotenv';
dotenv.config();

console.log("API Key loaded:", !!process.env.GEMINI_API_KEY);

import { embedder } from "./lib/embeddings.js";

try {
    console.log("Calling embedQuery...");
    const result = await embedder.embedQuery("hello world");
    console.log("Embedding length:", result.length);
    console.log("First 5 values:", result.slice(0, 5));
} catch (err) {
    console.error("Embedder error:", err.message);
    console.error("Full error:", err);
}

console.log("Done");