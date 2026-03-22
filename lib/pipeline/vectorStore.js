import { QdrantVectorStore } from "@langchain/qdrant";
import { embedder } from "../embeddings.js";
import crypto from "crypto";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import { pipelineLog } from "../logger.js";

const INTER_BATCH_DELAY_MS = 250;  // spacing between batch requests
const RETRY_DELAY_MS = 4000;       // wait before retrying a failed batch
const MAX_RETRIES = 3;
const BATCH_SIZE = parseInt(process.env.EMBEDDING_BATCH_SIZE || "20", 10);
const BATCH_CONCURRENCY = parseInt(process.env.EMBEDDING_BATCH_CONCURRENCY || "1", 10);
const CACHE_DIR = path.join(process.cwd(), ".cache");
const EMBEDDING_CACHE_PATH = path.join(CACHE_DIR, "embedding-cache.json");

let embeddingCache = null;

function sanitizeCollectionName(name) {
    const safe = (name || "my_collection")
        .toLowerCase()
        .replace(/[^a-z0-9_-]/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "");
    return safe || "my_collection";
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function ensureCacheLoaded() {
    if (embeddingCache) return embeddingCache;
    try {
        if (!existsSync(CACHE_DIR)) await fs.mkdir(CACHE_DIR, { recursive: true });
        if (existsSync(EMBEDDING_CACHE_PATH)) {
            const raw = await fs.readFile(EMBEDDING_CACHE_PATH, "utf-8");
            embeddingCache = new Map(Object.entries(JSON.parse(raw || "{}")));
        } else {
            embeddingCache = new Map();
        }
    } catch (err) {
        pipelineLog.warn({ err: err.message }, 'Failed to load embedding cache, using in-memory only');
        embeddingCache = new Map();
    }
    return embeddingCache;
}

async function persistCache() {
    if (!embeddingCache) return;
    try {
        const obj = Object.fromEntries(embeddingCache);
        // Atomic write: write to temp file then rename
        const tmpPath = EMBEDDING_CACHE_PATH + '.tmp';
        await fs.writeFile(tmpPath, JSON.stringify(obj));
        await fs.rename(tmpPath, EMBEDDING_CACHE_PATH);
    } catch (err) {
        pipelineLog.warn({ err: err.message }, 'Failed to persist embedding cache');
    }
}

function hashText(text) {
    return crypto
        .createHash("sha256")
        .update((text || "").replace(/\s+/g, " ").trim())
        .digest("hex");
}

async function runWithConcurrency(items, worker, concurrency = 1) {
    const limit = Math.max(1, concurrency);
    let index = 0;
    const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
        while (true) {
            const current = index++;
            if (current >= items.length) break;
            await worker(items[current], current);
        }
    });
    await Promise.all(workers);
}

function isRetryable(err) {
    if (err?.status) return err.status === 429 || err.status >= 500;
    const msg = (err?.message || "").toLowerCase();
    return msg.includes("rate") || msg.includes("timeout") || msg.includes("temporar");
}

function normalizeEmbeddingText(text) {
    return (text || "")
        .replace(/\u0000/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

async function embedSingleWithRetry(base, text, batchNumber, itemNumber) {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const vec = await base.embedQuery(text);
            if (!vec || vec.length === 0) {
                throw new Error(`Empty vector for batch #${batchNumber}, item #${itemNumber}`);
            }
            return vec;
        } catch (err) {
            const retryable = isRetryable(err) || (err?.message || "").toLowerCase().includes("empty vector");
            pipelineLog.warn(
                `    Item #${itemNumber} attempt ${attempt} failed: ${err.message}${retryable && attempt < MAX_RETRIES ? `, retrying in ${RETRY_DELAY_MS}ms...` : ""}`
            );
            if (!retryable || attempt === MAX_RETRIES) throw err;
            await sleep(RETRY_DELAY_MS);
        }
    }
    throw new Error(`Failed embedding item #${itemNumber} after ${MAX_RETRIES} retries.`);
}

async function embedBatchWithRetry(base, texts, batchNumber) {
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const vectors = await base.embedDocuments(texts);
            if (!Array.isArray(vectors) || vectors.length !== texts.length) {
                throw new Error(`Vector count mismatch for batch #${batchNumber}`);
            }
            for (let i = 0; i < vectors.length; i++) {
                if (!vectors[i] || vectors[i].length === 0) {
                    throw new Error(`Empty vector at batch #${batchNumber}, item #${i + 1}`);
                }
            }
            return vectors;
        } catch (err) {
            const msg = (err?.message || "").toLowerCase();
            const retryable = isRetryable(err) || msg.includes("empty vector") || msg.includes("vector count mismatch");
            pipelineLog.warn(
                `  Batch #${batchNumber} attempt ${attempt} failed: ${err.message}${retryable && attempt < MAX_RETRIES ? `, retrying in ${RETRY_DELAY_MS}ms...` : ""}`
            );
            if (!retryable || attempt === MAX_RETRIES) throw err;
            await sleep(RETRY_DELAY_MS);
        }
    }
    throw new Error(`Failed embedding batch #${batchNumber} after ${MAX_RETRIES} retries.`);
}

/**
 * Embeds texts in batches, reuses cached embeddings by chunk hash,
 * and retries failed batches with backoff.
 */
class BatchCachingEmbedder {
    constructor(base) { this.base = base; }

    async embedDocuments(texts) {
        const cache = await ensureCacheLoaded();
        const results = new Array(texts.length);
        const toEmbed = [];
        let cacheHits = 0;

        for (let i = 0; i < texts.length; i++) {
            const hash = hashText(texts[i]);
            const cached = cache.get(hash);
            if (cached) {
                results[i] = cached;
                cacheHits++;
            } else {
                toEmbed.push({ idx: i, hash, text: texts[i] });
            }
        }

        if (toEmbed.length > 0) {
            const batches = [];
            for (let i = 0; i < toEmbed.length; i += BATCH_SIZE) {
                batches.push(toEmbed.slice(i, i + BATCH_SIZE));
            }

            pipelineLog.info({ total: texts.length, cacheHits, toEmbed: toEmbed.length, batches: batches.length, batchSize: BATCH_SIZE }, 'Embedding documents');

            await runWithConcurrency(
                batches,
                async (batch, batchIdx) => {
                    const batchNumber = batchIdx + 1;
                    const batchTexts = batch.map(x => normalizeEmbeddingText(x.text));
                    try {
                        const vectors = await embedBatchWithRetry(this.base, batchTexts, batchNumber);
                        for (let i = 0; i < batch.length; i++) {
                            const item = batch[i];
                            const vec = vectors[i];
                            results[item.idx] = vec;
                            cache.set(item.hash, vec);
                        }
                    } catch (batchErr) {
                        pipelineLog.warn(`  Batch #${batchNumber} fallback: isolating items due to "${batchErr.message}"`);
                        for (let i = 0; i < batch.length; i++) {
                            const item = batch[i];
                            const text = batchTexts[i];
                            try {
                                const vec = await embedSingleWithRetry(this.base, text, batchNumber, i + 1);
                                results[item.idx] = vec;
                                cache.set(item.hash, vec);
                            } catch (itemErr) {
                                // Last-resort safety: skip a truly unembeddable chunk rather than failing whole upload.
                                pipelineLog.warn(`    Skipping unembeddable item #${i + 1} in batch #${batchNumber}: ${itemErr.message}`);
                                results[item.idx] = null;
                            }
                        }
                    }
                    await sleep(INTER_BATCH_DELAY_MS);
                },
                BATCH_CONCURRENCY
            );

            await persistCache();
        }

        const firstValid = results.find(v => Array.isArray(v) && v.length > 0);
        if (!firstValid) {
            throw new Error("Embedding generation failed for all chunks.");
        }
        const embeddingDim = firstValid.length;
        for (let i = 0; i < results.length; i++) {
            if (!results[i] || results[i].length === 0) {
                // Keep pipeline alive by assigning a deterministic zero-vector placeholder.
                // This chunk becomes effectively non-retrievable but avoids ingestion failure.
                results[i] = new Array(embeddingDim).fill(0);
            }
        }

        return results;
    }

    // For retrieval queries we keep direct model calls (usually short text).
    async embedQuery(text) {
        return this.base.embedQuery(text);
    }
}

export async function storeDocuments(docs, collectionName = "my_collection") {
    if (!docs || docs.length === 0) {
        throw new Error("No documents provided to store in Qdrant.");
    }

    if (!process.env.QDRANT_URL) {
        throw new Error("QDRANT_URL is required. Set it in .env before uploading documents.");
    }
    if (!process.env.QDRANT_API_KEY) {
        throw new Error("QDRANT_API_KEY is required. Set it in .env before uploading documents.");
    }

    const qdrantCollection = sanitizeCollectionName(collectionName);
    pipelineLog.info(`Storing ${docs.length} document chunks in Qdrant collection "${qdrantCollection}" (batched + cache + retry)...`);

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

    const optimizedEmbedder = new BatchCachingEmbedder(embedder);

    // fromDocuments calls optimizedEmbedder.embedDocuments internally.
    // This gives batched embedding, retries, and persistent chunk-hash caching.
    const vectorStore = await QdrantVectorStore.fromDocuments(sanitizedDocs, optimizedEmbedder, {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: qdrantCollection,
    });

    pipelineLog.info(`All ${sanitizedDocs.length} chunks stored successfully in Qdrant.`);
    return vectorStore;
}

