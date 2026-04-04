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
 */
export async function restoreSession(sessionId) {
    try {
        const doc = await getDocumentBySessionId(sessionId);
        if (!doc) {
            serverLog.debug({ sessionId }, 'Session restore: no document found in DB');
            return null;
        }

        const collectionName = sanitizeCollectionName(doc.qdrant_collection || sessionId);
        const isImageSession = collectionName.endsWith('-img');

        // Restore chat history from DB
        let chatHistory = [];
        try {
            const dbHistory = await getChatHistory(doc.id, 20);
            chatHistory = dbHistory.map(m =>
                m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content)
            );
        } catch (err) {
            serverLog.warn({ err: err.message, sessionId }, 'Session restore: failed to load chat history');
        }

        // Image sessions: don't reconnect via LangChain (different embedding dims)
        if (isImageSession) {
            serverLog.info({ sessionId, collectionName, chatMessages: chatHistory.length }, 'Session restored from DB + Qdrant (image)');
            return {
                vectorStore: null,
                docs: [],
                chatHistory,
                interviewState: null,
                createdAt: Date.now(),
                _documentId: doc.id,
                _restored: true,
                originalName: doc.original_name,
                contentType: 'image',
                hasImages: true,
                imageCollection: collectionName,
            };
        }

        // Text/PDF sessions: reconnect to Qdrant via LangChain
        const vectorStore = await QdrantVectorStore.fromExistingCollection(embedder, {
            url: process.env.QDRANT_URL,
            apiKey: process.env.QDRANT_API_KEY,
            collectionName,
        });

        serverLog.info({ sessionId, collectionName, chatMessages: chatHistory.length }, 'Session restored from DB + Qdrant');

        return {
            vectorStore,
            docs: [],
            chatHistory,
            interviewState: null,
            createdAt: Date.now(),
            _documentId: doc.id,
            _restored: true,
            originalName: doc.original_name,
        };

    } catch (err) {
        serverLog.error({ err: err.message, sessionId }, 'Session restore failed');
        return null;
    }
}

/**
 * Ensures a session exists in sessionCache. If missing, attempts to restore from DB.
 * Returns the session object or null if unrestorable.
 */
export async function ensureSession(sessionCache, sessionId) {
    // Already in memory (text session with vectorStore OR image session)
    const cached = sessionCache[sessionId];
    if (cached && (cached.vectorStore || cached.contentType === 'image' || cached.contentType === 'text')) {
        return cached;
    }

    // Try to restore
    const restored = await restoreSession(sessionId);
    if (restored) {
        sessionCache[sessionId] = restored;
        return restored;
    }

    return null;
}
