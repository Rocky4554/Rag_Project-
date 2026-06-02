import { searchBM25 } from './bm25Index.js'
import { rerank } from './reranker.js'
import { chatLog } from '../logger.js'

const VECTOR_K = 50   // candidates from semantic search
const BM25_K = 50     // candidates from keyword search
const RRF_K = 60      // RRF constant (higher = smoother rank blending)
const RERANK_POOL = 20 // how many merged candidates to send to the reranker

/**
 * Reciprocal Rank Fusion — merges two ranked lists by rewarding docs
 * that appear high in both. Deduplicates by first-120-chars of content.
 */
function reciprocalRankFusion(vectorDocs, bm25Docs) {
    const scores = new Map()
    const docMap = new Map()

    const addList = (docs) => {
        for (let rank = 0; rank < docs.length; rank++) {
            const doc = docs[rank]
            const key = doc.pageContent.substring(0, 120)
            scores.set(key, (scores.get(key) || 0) + 1 / (RRF_K + rank + 1))
            if (!docMap.has(key)) docMap.set(key, doc)
        }
    }

    addList(vectorDocs)
    addList(bm25Docs)

    return [...scores.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([key]) => docMap.get(key))
        .filter(Boolean)
}

/**
 * Full hybrid retrieval pipeline:
 *   1. BM25 keyword search  ┐
 *   2. Semantic vector search┘  (parallel)
 *   3. Reciprocal Rank Fusion merge
 *   4. Jina Reranker (cross-encoder)
 *   5. Return top K for the LLM
 *
 * If bm25Index is null (session restored from DB without in-memory index),
 * falls back to semantic-only + reranker — still better than bare k:3.
 *
 * @param {object} vectorStore  - Qdrant vector store instance
 * @param {object|null} bm25Index - in-memory BM25 index from buildBM25Index()
 * @param {string} vectorQuery  - contextual query (may include history prefix) for semantic search
 * @param {string} rawQuery     - clean user question for BM25 keyword matching
 * @param {object} opts
 * @param {number} opts.topK    - final docs to return to LLM (default 10)
 */
export async function hybridRetrieve(vectorStore, bm25Index, vectorQuery, rawQuery, { topK = 10 } = {}) {
    const start = performance.now()

    const [vectorDocs, bm25Docs] = await Promise.all([
        vectorStore.asRetriever({ k: VECTOR_K }).invoke(vectorQuery),
        Promise.resolve(bm25Index ? searchBM25(bm25Index, rawQuery, BM25_K) : []),
    ])

    const retrieveMs = Math.round(performance.now() - start)
    chatLog.debug(
        { retrieveMs, vectorDocs: vectorDocs.length, bm25Docs: bm25Docs.length },
        'Parallel retrieval done'
    )

    let candidates
    if (bm25Docs.length === 0) {
        candidates = vectorDocs.slice(0, RERANK_POOL)
    } else {
        const merged = reciprocalRankFusion(vectorDocs, bm25Docs)
        candidates = merged.slice(0, RERANK_POOL)
        chatLog.debug({ mergedTotal: merged.length, sentToReranker: candidates.length }, 'RRF merge done')
    }

    const reranked = await rerank(rawQuery, candidates, topK)

    const totalMs = Math.round(performance.now() - start)
    chatLog.info(
        { totalMs, final: reranked.length, hadBM25: bm25Docs.length > 0 },
        'Hybrid retrieval complete'
    )

    return reranked
}
