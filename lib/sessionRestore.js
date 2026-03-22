import { QdrantVectorStore } from "@langchain/qdrant";
import { embedder } from "./embeddings.js";
import { getDocumentBySessionId, getChatHistory } from "./db.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { serverLog } from "./logger.js";

/**
 * Sanitize collection name the same way vectorStore.js does.
 */
function sanitizeCollectionName(name) {
    const safe = (name || "my_collection")
        .toLowerCase()
        .replace(/[^a-z0-9_-]/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "");
    return safe || "my_collection";
}

/**
 * Restore a session from Supabase DB + existing Qdrant collection.
 * Returns a hydrated session object ready to be placed in sessionCache,
 * or null if the session can't be restored (no DB record or Qdrant collection missing).
 *
 * @param {string} sessionId
 * @returns {Promise<object|null>} Hydrated session or null
 */
export async function restoreSession(sessionId) {
    try {
        // 1. Look up document in Supabase
        const doc = await getDocumentBySessionId(sessionId);
        if (!doc) {
            serverLog.debug({ sessionId }, 'Session restore: no document found in DB');
            return null;
        }

        // 2. Reconnect to existing Qdrant collection (no re-embedding needed)
        const collectionName = sanitizeCollectionName(doc.qdrant_collection || sessionId);

        const vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
            url: process.env.QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName,
        });

        // 3. Restore chat history from DB
        let chatHistory = [];
        try {
            const dbHistory = await getChatHistory(doc.id, 20);
            chatHistory = dbHistory.map(m =>
                m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content)
            );
        } catch (err) {
            serverLog.warn({ err: err.message, sessionId }, 'Session restore: failed to load chat history');
        }

        serverLog.info({ sessionId, collectionName, chatMessages: chatHistory.length }, 'Session restored from DB + Qdrant');

        return {
            vectorStore,
            docs: [],               // Original chunks not needed for querying — Qdrant has them
            chatHistory,
            interviewState: null,
            createdAt: Date.now(),   // Reset TTL
            _documentId: doc.id,
            _restored: true,         // Flag so we know this was restored
        };

    } catch (err) {
        serverLog.error({ err: err.message, sessionId }, 'Session restore failed');
        return null;
    }
}

/**
 * Ensures a session exists in sessionCache. If missing, attempts to restore from DB.
 * Returns the session object or null if unrestorable.
 *
 * @param {object} sessionCache - The global sessionCache object
 * @param {string} sessionId
 * @returns {Promise<object|null>}
 */
export async function ensureSession(sessionCache, sessionId) {
    // Already in memory
    if (sessionCache[sessionId]?.vectorStore) {
        return sessionCache[sessionId];
    }

    // Try to restore
    const restored = await restoreSession(sessionId);
    if (restored) {
        sessionCache[sessionId] = restored;
        return restored;
    }

    return null;
}
