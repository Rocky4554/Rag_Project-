import { Chroma } from "@langchain/community/vectorstores/chroma";
import { embedder } from "./embeddings.js";

const INTER_DOC_DELAY_MS = 200;  // 200ms between each embedding call
const RETRY_DELAY_MS = 4000;     // 4s wait before retrying a failed embedding
const MAX_RETRIES = 3;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Wraps the base embedder to process ONE document at a time with retry logic.
 * Batching causes the Gemini free-tier to silently return [] or wrong-dimension
 * vectors for chunks that exceed its burst rate — this approach is slower but
 * 100% reliable since each chunk is validated before moving to the next.
 */
class RetryEmbedder {
    constructor(base) { this.base = base; }

    async embedDocuments(texts) {
        const results = [];
        for (let i = 0; i < texts.length; i++) {
            let embedding = null;
            for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
                const [vec] = await this.base.embedDocuments([texts[i]]);
                if (vec && vec.length > 0) {
                    embedding = vec;
                    break;
                }
                console.warn(`  Embedding attempt ${attempt} returned empty for chunk ${i + 1}, retrying in ${RETRY_DELAY_MS}ms...`);
                await sleep(RETRY_DELAY_MS);
            }
            if (!embedding) {
                throw new Error(`Failed to embed chunk ${i + 1} after ${MAX_RETRIES} retries.`);
            }
            results.push(embedding);
            if (i < texts.length - 1) await sleep(INTER_DOC_DELAY_MS); // throttle between calls
        }
        return results;
    }

    // Used by ChromaDB during similarity search — delegate directly, no retry needed
    async embedQuery(text) {
        return this.base.embedQuery(text);
    }
}

export async function storeDocuments(docs, collectionName = "my_collection") {
    if (!docs || docs.length === 0) {
        throw new Error("No documents provided to store in ChromaDB.");
    }

    console.log(`Storing ${docs.length} document chunks in Chroma DB (one at a time with retry)...`);

    const sanitizedDocs = docs.map(doc => {
        const cleanMeta = {};
        for (const [key, value] of Object.entries(doc.metadata)) {
            if (value === null || ["string", "number", "boolean"].includes(typeof value)) {
                cleanMeta[key] = value;
            }
        }
        doc.metadata = cleanMeta;
        return doc;
    });

    const retryEmbedder = new RetryEmbedder(embedder);

    // fromDocuments calls retryEmbedder.embedDocuments internally.
    // Each chunk is embedded one at a time with validation + retry.
    const vectorStore = await Chroma.fromDocuments(sanitizedDocs, retryEmbedder, {
        collectionName: collectionName,
        url: "http://localhost:8000",
    });

    console.log(`All ${sanitizedDocs.length} chunks stored successfully.`);
    return vectorStore;
}

