import { QdrantVectorStore } from "@langchain/qdrant";
import { embedder } from "../embeddings.js";
import crypto from "crypto";
import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import { pipelineLog, embeddingLog } from "../logger.js";

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
            // Keep only the newest entries if the file predates the size cap
            embeddingCache = new Map(Object.entries(JSON.parse(raw || "{}")).slice(-CACHE_MAX_ENTRIES));
        } else {
            embeddingCache = new Map();
        }
    } catch (err) {
        pipelineLog.warn({ err: err.message }, 'Failed to load embedding cache, using in-memory only');
        embeddingCache = new Map();
    }
    return embeddingCache;
}

// Each entry is a full embedding (~3k floats ≈ 60KB of JSON), so the cache is
// capped and evicts least-recently-used entries (Map keeps insertion order).
const CACHE_MAX_ENTRIES = parseInt(process.env.EMBEDDING_CACHE_MAX_ENTRIES || "500", 10);

function cacheGet(cache, key) {
    const vec = cache.get(key);
    if (vec) {
        cache.delete(key);
        cache.set(key, vec); // mark as most recently used
    }
    return vec;
}

function cacheSet(cache, key, vec) {
    cache.delete(key);
    cache.set(key, vec);
    while (cache.size > CACHE_MAX_ENTRIES) {
        cache.delete(cache.keys().next().value);
    }
}

// Writes run one at a time: concurrent uploads would otherwise race on the temp file.
let persistChain = Promise.resolve();

function persistCache() {
    persistChain = persistChain.then(async () => {
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
    });
    return persistChain;
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
        const embedStart = performance.now();
        const cache = await ensureCacheLoaded();
        const results = new Array(texts.length);
        const toEmbed = [];
        let cacheHits = 0;

        for (let i = 0; i < texts.length; i++) {
            const hash = hashText(texts[i]);
            const cached = cacheGet(cache, hash);
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

            embeddingLog.info({ total: texts.length, cacheHits, toEmbed: toEmbed.length, batches: batches.length, batchSize: BATCH_SIZE }, 'Embedding documents');

            await runWithConcurrency(
                batches,
                async (batch, batchIdx) => {
                    const batchNumber = batchIdx + 1;
                    const batchTexts = batch.map(x => normalizeEmbeddingText(x.text));
                    const batchStart = performance.now();
                    try {
                        const vectors = await embedBatchWithRetry(this.base, batchTexts, batchNumber);
                        const batchMs = Math.round(performance.now() - batchStart);
                        embeddingLog.info({ batchNumber, batchSize: batch.length, durationMs: batchMs }, 'Batch embedded');
                        for (let i = 0; i < batch.length; i++) {
                            const item = batch[i];
                            const vec = vectors[i];
                            results[item.idx] = vec;
                            cacheSet(cache, item.hash, vec);
                        }
                    } catch (batchErr) {
                        const batchMs = Math.round(performance.now() - batchStart);
                        embeddingLog.warn({ batchNumber, durationMs: batchMs, err: batchErr.message }, 'Batch failed, isolating items');
                        for (let i = 0; i < batch.length; i++) {
                            const item = batch[i];
                            const text = batchTexts[i];
                            try {
                                const vec = await embedSingleWithRetry(this.base, text, batchNumber, i + 1);
                                results[item.idx] = vec;
                                cacheSet(cache, item.hash, vec);
                            } catch (itemErr) {
                                embeddingLog.warn({ batchNumber, itemNumber: i + 1, err: itemErr.message }, 'Skipping unembeddable item');
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
        const failedCount = results.filter(v => !v || v.length === 0).length;

        const totalMs = Math.round(performance.now() - embedStart);
        embeddingLog.info(
            { totalMs, total: texts.length, cacheHits, embedded: toEmbed.length, failed: failedCount, embeddingDim: firstValid.length },
            'All embeddings complete'
        );

        // Failed items are left as null; storeDocuments drops them rather than storing zero vectors.
        return results;
    }

    // For retrieval queries we keep direct model calls (usually short text).
    async embedQuery(text) {
        const start = performance.now();
        const result = await this.base.embedQuery(text);
        const durationMs = Math.round(performance.now() - start);
        embeddingLog.debug({ durationMs, textLength: text.length }, 'Query embedded');
        return result;
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

    const storeStart = performance.now();
    const qdrantCollection = sanitizeCollectionName(collectionName);
    pipelineLog.info({ chunks: docs.length, collection: qdrantCollection }, 'Storing documents in Qdrant');

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

    // Embed explicitly (batched, retried, chunk-hash cached) so chunks that
    // failed to embed can be dropped instead of stored as zero vectors.
    const vectors = await optimizedEmbedder.embedDocuments(sanitizedDocs.map(d => d.pageContent));
    const keptDocs = [];
    const keptVectors = [];
    vectors.forEach((vec, i) => {
        if (vec && vec.length > 0) {
            keptDocs.push(sanitizedDocs[i]);
            keptVectors.push(vec);
        }
    });

    const vectorStore = new QdrantVectorStore(optimizedEmbedder, {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: qdrantCollection,
    });
    await vectorStore.addVectors(keptVectors, keptDocs);

    const storeMs = Math.round(performance.now() - storeStart);
    pipelineLog.info(
        { chunks: keptDocs.length, dropped: sanitizedDocs.length - keptDocs.length, collection: qdrantCollection, durationMs: storeMs },
        'All chunks stored in Qdrant'
    );
    return vectorStore;
}

