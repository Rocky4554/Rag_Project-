import dotenv from 'dotenv';
dotenv.config();
import { embedder } from "./lib/embeddings.js";

async function main() {
    try {
        console.log("Calling embedDocuments...");
        const result = await embedder.embedDocuments(["hello world"]);
        console.log("Result length:", result.length);
        console.log("Result:", result);
    } catch (err) {
        console.error("Embedder error:", err.message);
        console.error("Full error:", err);
    }
}
main();
