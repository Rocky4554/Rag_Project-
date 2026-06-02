import { chatLog } from '../logger.js'

/**
 * Rerank candidate documents against a query using Jina Reranker v2.
 * Falls back to returning docs.slice(0, topK) if the API key is missing or the call fails.
 */
export async function rerank(query, docs, topK = 10) {
    if (!process.env.JINA_API_KEY || docs.length === 0) {
        return docs.slice(0, topK)
    }

    const start = performance.now()
    try {
        const response = await fetch('https://api.jina.ai/v1/rerank', {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${process.env.JINA_API_KEY}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'jina-reranker-v2-base-multilingual',
                query,
                documents: docs.map(d => d.pageContent),
                top_n: topK,
            }),
        })

        if (!response.ok) {
            throw new Error(`Reranker HTTP ${response.status}: ${response.statusText}`)
        }

        const { results } = await response.json()
        const ms = Math.round(performance.now() - start)
        chatLog.info({ ms, input: docs.length, output: results.length }, 'Rerank complete')

        return results.map(r => docs[r.index])
    } catch (err) {
        const ms = Math.round(performance.now() - start)
        chatLog.warn({ ms, err: err.message }, 'Reranker failed, using top K candidates instead')
        return docs.slice(0, topK)
    }
}
