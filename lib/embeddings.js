import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { JinaEmbeddings } from "@langchain/community/embeddings/jina";
import dotenv from 'dotenv';
import { embeddingLog } from './logger.js';

dotenv.config();

/**
 * Embedding provider chain with automatic fallback:
 *   1. gemini-embedding-001    (Google, 3072 dims)
 *   2. gemini-embedding-2-preview (Google, 3072 dims)
 *   3. jina-embeddings-v3      (Jina AI, 1024 dims)
 *
 * IMPORTANT: Within a single Qdrant collection all vectors must have the same
 * dimension. The fallback is designed for when the primary provider is DOWN
 * (rate limit, quota exhausted, outage). If Jina kicks in mid-upload its
 * different dimension will cause a Qdrant error — this is intentional so
 * you know the primary failed, rather than silently mixing dimensions.
 * For fresh uploads with a fully-failed Gemini, Jina will create a new
 * collection with its own dimension and work fine.
 */

const providers = [
    {
        name: 'gemini-embedding-001',
        create: () => new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            model: "gemini-embedding-001",
        }),
        available: !!process.env.GEMINI_API_KEY,
    },
    {
        name: 'gemini-embedding-2-preview',
        create: () => new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            model: "gemini-embedding-2-preview",
        }),
        available: !!process.env.GEMINI_API_KEY,
    },
    {
        name: 'jina-embeddings-v3',
        create: () => new JinaEmbeddings({
            apiKey: process.env.JINA_API_KEY,
            model: "jina-embeddings-v3",
        }),
        available: !!process.env.JINA_API_KEY,
    },
];

const availableProviders = providers.filter(p => p.available);

if (availableProviders.length === 0) {
    embeddingLog.fatal('No embedding API keys configured (need GEMINI_API_KEY or JINA_API_KEY)');
    throw new Error('No embedding providers available');
}

embeddingLog.info(
    { providers: availableProviders.map(p => p.name), primary: availableProviders[0].name },
    'Embedding fallback chain initialized'
);

// Max time to wait for a single embedding provider before trying the next one
const PROVIDER_TIMEOUT_MS = 15_000;

/**
 * Races an embedding call against a timeout so the fallback kicks in fast
 * instead of waiting 90+ seconds for the SDK's internal retries.
 */
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
 * Implements the same interface as LangChain Embeddings (embedQuery, embedDocuments).
 *
 * Circuit breaker: when a provider hits a daily quota error, it is skipped
 * for subsequent calls until the quota resets (checked every 10 minutes).
 */
class FallbackEmbeddings {
    constructor(providerConfigs) {
        this._providers = providerConfigs;
        // Lazily instantiated — only create when first needed
        this._instances = new Array(providerConfigs.length).fill(null);
        // Circuit breaker: tracks when a provider is tripped (daily quota)
        // Map<index, tripTimestamp>
        this._tripped = new Map();
    }

    _getInstance(idx) {
        if (!this._instances[idx]) {
            this._instances[idx] = this._providers[idx].create();
        }
        return this._instances[idx];
    }

    /** Returns true if provider should be skipped (tripped within last 10 min) */
    _isTripped(idx) {
        const tripTime = this._tripped.get(idx);
        if (!tripTime) return false;
        // Re-check every 10 minutes in case quota resets
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

                // Trip circuit breaker on daily quota exhaustion
                if (rateLimitType === 'quota_exhausted') {
                    this._trip(i, provider);
                }

                embeddingLog.error(
                    { provider: provider.name, durationMs, err: err.message, rateLimitType, texts: texts.length, hasNext: !isLast },
                    isLast ? 'All embedding providers failed (embedDocuments)' : 'Embedding provider failed, trying next'
                );

                if (isLast) throw err;
                // Reset instance so next attempt creates a fresh one
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

                if (rateLimitType === 'quota_exhausted') {
                    this._trip(i, provider);
                }

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

export const embedder = new FallbackEmbeddings(availableProviders);
