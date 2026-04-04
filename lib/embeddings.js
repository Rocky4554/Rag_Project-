import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { JinaEmbeddings } from "@langchain/community/embeddings/jina";
import dotenv from 'dotenv';
import { embeddingLog } from './logger.js';

dotenv.config();

/**
 * Embedding provider chain with key rotation and automatic fallback:
 *
 *   For text:
 *     1. gemini-embedding-2-preview (rotates through all GEMINI_API_KEYS)
 *     2. gemini-embedding-001       (rotates through all GEMINI_API_KEYS)
 *     3. jina-embeddings-v3         (Jina AI, 1024 dims)
 *
 *   For images:
 *     → jina-clip-v2 (multimodal, 1024 dims)
 *
 * Each Gemini model is tried with every available API key before moving
 * to the next model. This maximizes availability when individual keys
 * hit rate limits or quota exhaustion.
 */

// ── Parse Gemini keys ──────────────────────────────────────────────
const geminiKeys = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
    .split(',')
    .map(k => k.trim())
    .filter(Boolean);

const jinaKey = process.env.JINA_API_KEY;

// ── Build text embedding provider chain ────────────────────────────
// Order: gemini-2-preview (all keys) → gemini-001 (all keys) → jina-v3
const providers = [];

for (const model of ['gemini-embedding-2-preview', 'gemini-embedding-001']) {
    for (let i = 0; i < geminiKeys.length; i++) {
        const key = geminiKeys[i];
        providers.push({
            name: `${model}/key-${i + 1}`,
            create: () => new GoogleGenerativeAIEmbeddings({ apiKey: key, model }),
            available: true,
        });
    }
}

if (jinaKey) {
    providers.push({
        name: 'jina-embeddings-v3',
        create: () => new JinaEmbeddings({ apiKey: jinaKey, model: "jina-embeddings-v3" }),
        available: true,
    });
}

const availableProviders = providers.filter(p => p.available);

if (availableProviders.length === 0) {
    embeddingLog.fatal('No embedding API keys configured (need GEMINI_API_KEYS or JINA_API_KEY)');
    throw new Error('No embedding providers available');
}

embeddingLog.info(
    { providers: availableProviders.map(p => p.name), primary: availableProviders[0].name },
    'Embedding fallback chain initialized'
);

// Max time to wait for a single embedding provider before trying the next one
const PROVIDER_TIMEOUT_MS = 15_000;

function withTimeout(promise, ms) {
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`Embedding provider timeout (${ms}ms)`)), ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * FallbackEmbeddings wraps multiple LangChain embedding providers.
 * On any error it logs the failure and tries the next provider in the chain.
 * Circuit breaker: when a provider hits a daily quota error, it is skipped
 * for subsequent calls until the quota resets (checked every 10 minutes).
 */
class FallbackEmbeddings {
    constructor(providerConfigs) {
        this._providers = providerConfigs;
        this._instances = new Array(providerConfigs.length).fill(null);
        this._tripped = new Map();
    }

    _getInstance(idx) {
        if (!this._instances[idx]) {
            this._instances[idx] = this._providers[idx].create();
        }
        return this._instances[idx];
    }

    _isTripped(idx) {
        const tripTime = this._tripped.get(idx);
        if (!tripTime) return false;
        if (Date.now() - tripTime > 10 * 60 * 1000) {
            this._tripped.delete(idx);
            return false;
        }
        return true;
    }

    _trip(idx, provider) {
        this._tripped.set(idx, Date.now());
        embeddingLog.warn({ provider: provider.name }, 'Circuit breaker tripped — skipping provider for 10 min');
    }

    async embedDocuments(texts) {
        for (let i = 0; i < this._providers.length; i++) {
            const provider = this._providers[i];
            if (this._isTripped(i)) {
                embeddingLog.info({ provider: provider.name }, 'Skipping tripped provider (circuit breaker)');
                if (i === this._providers.length - 1) throw new Error('All embedding providers are circuit-broken');
                continue;
            }
            const start = performance.now();
            try {
                const instance = this._getInstance(i);
                const result = await withTimeout(instance.embedDocuments(texts), PROVIDER_TIMEOUT_MS);
                const durationMs = Math.round(performance.now() - start);
                if (i > 0) {
                    embeddingLog.info({ provider: provider.name, durationMs, texts: texts.length }, 'Fallback embedDocuments succeeded');
                }
                return result;
            } catch (err) {
                const durationMs = Math.round(performance.now() - start);
                const isLast = i === this._providers.length - 1;
                const rateLimitType = detectEmbeddingRateLimit(err);
                if (rateLimitType === 'quota_exhausted') this._trip(i, provider);

                embeddingLog.error(
                    { provider: provider.name, durationMs, err: err.message, rateLimitType, texts: texts.length, hasNext: !isLast },
                    isLast ? 'All embedding providers failed (embedDocuments)' : 'Embedding provider failed, trying next'
                );

                if (isLast) throw err;
                this._instances[i] = null;
            }
        }
    }

    async embedQuery(text) {
        for (let i = 0; i < this._providers.length; i++) {
            const provider = this._providers[i];
            if (this._isTripped(i)) {
                embeddingLog.info({ provider: provider.name }, 'Skipping tripped provider (circuit breaker)');
                if (i === this._providers.length - 1) throw new Error('All embedding providers are circuit-broken');
                continue;
            }
            const start = performance.now();
            try {
                const instance = this._getInstance(i);
                const result = await withTimeout(instance.embedQuery(text), PROVIDER_TIMEOUT_MS);
                const durationMs = Math.round(performance.now() - start);
                if (i > 0) {
                    embeddingLog.info({ provider: provider.name, durationMs }, 'Fallback embedQuery succeeded');
                }
                return result;
            } catch (err) {
                const durationMs = Math.round(performance.now() - start);
                const isLast = i === this._providers.length - 1;
                const rateLimitType = detectEmbeddingRateLimit(err);
                if (rateLimitType === 'quota_exhausted') this._trip(i, provider);

                embeddingLog.error(
                    { provider: provider.name, durationMs, err: err.message, rateLimitType, hasNext: !isLast },
                    isLast ? 'All embedding providers failed (embedQuery)' : 'Embedding provider failed, trying next'
                );

                if (isLast) throw err;
                this._instances[i] = null;
            }
        }
    }
}

function detectEmbeddingRateLimit(err) {
    if (!err) return null;
    const msg = (err.message || '').toLowerCase();
    const status = err.status || err.statusCode || err.response?.status;
    if (status === 429) return 'rate_limit';
    if (status === 503) return 'overloaded';
    if (msg.includes('rate') && msg.includes('limit')) return 'rate_limit';
    if (msg.includes('quota') || msg.includes('exhausted')) return 'quota_exhausted';
    if (msg.includes('resource_exhausted') || msg.includes('resource exhausted')) return 'quota_exhausted';
    return null;
}

// ── Text embedder (main, used everywhere) ──────────────────────────
export const embedder = new FallbackEmbeddings(availableProviders);

// ── Image embedder (Jina CLIP v2, for image uploads) ───────────────
// Uses Jina's multimodal CLIP model that can embed both images and text
// into the same vector space. Returns 1024-dim vectors.
class JinaClipEmbedder {
    constructor(apiKey) {
        this._apiKey = apiKey;
        this._url = 'https://api.jina.ai/v1/embeddings';
    }

    /**
     * Embed one or more images (as base64 strings or URLs) via Jina CLIP v2.
     * @param {Array<{image?: string, text?: string}>} inputs
     * @returns {Promise<number[][]>} array of embedding vectors
     */
    async embed(inputs) {
        const start = performance.now();
        const resp = await fetch(this._url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this._apiKey}`,
            },
            body: JSON.stringify({ model: 'jina-clip-v2', input: inputs }),
        });

        if (!resp.ok) {
            const body = await resp.text();
            throw new Error(`Jina CLIP API error ${resp.status}: ${body}`);
        }

        const data = await resp.json();
        const durationMs = Math.round(performance.now() - start);
        embeddingLog.info({ provider: 'jina-clip-v2', items: inputs.length, durationMs }, 'Image embedding complete');
        return data.data.map(d => d.embedding);
    }

    /** Embed a single image (base64 string without data URI prefix) */
    async embedImage(base64) {
        const vectors = await this.embed([{ image: base64 }]);
        return vectors[0];
    }

    /** Embed a single image URL */
    async embedImageUrl(url) {
        const vectors = await this.embed([{ image: url }]);
        return vectors[0];
    }

    /** Embed text in the same CLIP vector space (for cross-modal search) */
    async embedQuery(text) {
        const vectors = await this.embed([{ text }]);
        return vectors[0];
    }
}

export const imageEmbedder = jinaKey ? new JinaClipEmbedder(jinaKey) : null;
