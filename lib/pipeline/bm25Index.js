import MiniSearch from 'minisearch'
import fs from 'fs/promises'
import { existsSync } from 'fs'
import path from 'path'
import { Document } from '@langchain/core/documents'

const BM25_CACHE_DIR = path.join(process.cwd(), '.cache', 'bm25')

function bm25CachePath(sessionId) {
    const safe = (sessionId || 'session')
        .replace(/[^a-z0-9_-]/gi, '_')
        .substring(0, 200)
    return path.join(BM25_CACHE_DIR, `bm25-${safe}.json`)
}

/**
 * Build an in-memory BM25 index from LangChain Document chunks.
 * Returns an opaque index object to be stored in sessionCache.
 */
export function buildBM25Index(docs) {
    const miniSearch = new MiniSearch({
        fields: ['content'],
        storeFields: ['docIdx'],
        searchOptions: {
            boost: { content: 1 },
            fuzzy: 0.1,
        },
    })

    miniSearch.addAll(
        docs.map((doc, i) => ({
            id: i,
            docIdx: i,
            content: doc.pageContent,
        }))
    )

    return { miniSearch, docs }
}

/**
 * Search the BM25 index and return matching LangChain Documents.
 * Gracefully returns [] if the index is missing or the query throws.
 */
export function searchBM25(bm25Index, query, topK = 50) {
    if (!bm25Index) return []
    const { miniSearch, docs } = bm25Index
    try {
        const results = miniSearch.search(query, { limit: topK })
        return results.map(r => docs[r.docIdx]).filter(Boolean)
    } catch {
        return []
    }
}

/**
 * Persist a BM25 index to disk so it survives server restarts.
 * Saves to .cache/bm25/bm25-{sessionId}.json (atomic write via temp file).
 * Fire-and-forget safe — errors are silently swallowed.
 */
export async function persistBM25Index(bm25Index, sessionId) {
    try {
        await fs.mkdir(BM25_CACHE_DIR, { recursive: true })
        const payload = JSON.stringify({
            index: JSON.parse(JSON.stringify(bm25Index.miniSearch)), // calls toJSON()
            docs: bm25Index.docs.map(d => ({
                pageContent: d.pageContent,
                metadata: d.metadata,
            })),
        })
        const filePath = bm25CachePath(sessionId)
        await fs.writeFile(filePath + '.tmp', payload, 'utf-8')
        await fs.rename(filePath + '.tmp', filePath)
    } catch {
        // Non-fatal: if persist fails, session restore falls back to semantic-only
    }
}

/**
 * Load a previously persisted BM25 index from disk.
 * Returns a hydrated { miniSearch, docs } object or null if not found.
 */
export async function loadBM25Index(sessionId) {
    const filePath = bm25CachePath(sessionId)
    if (!existsSync(filePath)) return null
    try {
        const raw = await fs.readFile(filePath, 'utf-8')
        const { index, docs: rawDocs } = JSON.parse(raw)
        const miniSearch = MiniSearch.loadJSON(JSON.stringify(index), {
            fields: ['content'],
            storeFields: ['docIdx'],
            searchOptions: { boost: { content: 1 }, fuzzy: 0.1 },
        })
        const docs = rawDocs.map(d => new Document({
            pageContent: d.pageContent,
            metadata: d.metadata,
        }))
        return { miniSearch, docs }
    } catch {
        return null
    }
}

/**
 * Delete the BM25 cache file for a single session.
 * Called when a session is evicted from the in-memory sessionCache.
 */
export async function deleteBM25File(sessionId) {
    try {
        await fs.unlink(bm25CachePath(sessionId))
    } catch {
        // File already gone or never created — not an error
    }
}

/**
 * Scan .cache/bm25/ and delete files older than maxAgeDays.
 * Run on server startup and every 24 hours to keep disk usage bounded.
 */
export async function cleanupOldBM25Files(maxAgeDays = 7) {
    try {
        if (!existsSync(BM25_CACHE_DIR)) return
        const entries = await fs.readdir(BM25_CACHE_DIR)
        const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000
        let deleted = 0
        for (const entry of entries) {
            if (!entry.endsWith('.json')) continue
            const filePath = path.join(BM25_CACHE_DIR, entry)
            try {
                const { mtimeMs } = await fs.stat(filePath)
                if (mtimeMs < cutoff) {
                    await fs.unlink(filePath)
                    deleted++
                }
            } catch {
                // Skip files we can't stat/delete
            }
        }
        if (deleted > 0) {
            console.info(`[bm25] Cleaned up ${deleted} stale index file(s) from .cache/bm25/`)
        }
    } catch {
        // Non-fatal
    }
}
