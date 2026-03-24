import { Router } from 'express';
import { routeQuery, routeQueryStream } from "../lib/pipeline/queryRouter.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { optionalAuth } from "../middleware/auth.js";
import { saveChatMessage, getChatHistory, getDocumentBySessionId } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { validate, chatSchema } from '../lib/validation.js';
import { chatLog } from '../lib/logger.js';

export function createChatRoutes({ sessionCache }) {
    const router = Router();

    router.post('/chat', validate(chatSchema), optionalAuth, async (req, res) => {
        const routeStart = performance.now();

        try {
            const { sessionId, question } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
            }

            chatLog.info({ sessionId, question: question.substring(0, 80), historySize: session.chatHistory.length }, 'Chat request received');

            // If user is authenticated and session chatHistory is empty, restore from DB
            if (req.user && session.chatHistory.length === 0) {
                try {
                    const doc = await getDocumentBySessionId(sessionId);
                    if (doc) {
                        const dbHistory = await getChatHistory(doc.id, 20);
                        session.chatHistory = dbHistory.map(m =>
                            m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content)
                        );
                        session._documentId = doc.id;
                        chatLog.info({ sessionId, restoredMessages: dbHistory.length }, 'Chat history restored from DB');
                    }
                } catch (err) {
                    chatLog.warn({ err: err.message }, 'Failed to restore chat history');
                }
            }

            const ragStart = performance.now();
            const { route, answer } = await routeQuery(session, question, session.chatHistory, req.user?.name);
            const ragMs = Math.round(performance.now() - ragStart);

            session.chatHistory.push(new HumanMessage(question));
            session.chatHistory.push(new AIMessage(answer));

            if (session.chatHistory.length > 20) {
                session.chatHistory = session.chatHistory.slice(session.chatHistory.length - 20);
            }

            // Persist chat messages to Supabase if user is authenticated
            if (req.user && session._documentId) {
                saveChatMessage({ userId: req.user.id, documentId: session._documentId, role: 'user', content: question });
                saveChatMessage({ userId: req.user.id, documentId: session._documentId, role: 'ai', content: answer });
            }

            const totalMs = Math.round(performance.now() - routeStart);
            chatLog.info(
                { sessionId, totalMs, ragMs, route, answerLength: answer.length, question: question.substring(0, 60) },
                'Chat response sent'
            );

            res.json({ answer, route });

        } catch (error) {
            const totalMs = Math.round(performance.now() - routeStart);
            chatLog.error({ err: error.message, totalMs }, 'Chat error');
            res.status(500).json({ error: error.message || "Failed to get answer" });
        }
    });

    // ── Streaming chat endpoint (SSE) ──────────────────────────────
    router.post('/chat/stream', validate(chatSchema), optionalAuth, async (req, res) => {
        const routeStart = performance.now();

        try {
            const { sessionId, question } = req.validated;

            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
            }

            chatLog.info({ sessionId, question: question.substring(0, 80), streaming: true }, 'Chat stream request received');

            // Restore chat history from DB if needed
            if (req.user && session.chatHistory.length === 0) {
                try {
                    const doc = await getDocumentBySessionId(sessionId);
                    if (doc) {
                        const dbHistory = await getChatHistory(doc.id, 20);
                        session.chatHistory = dbHistory.map(m =>
                            m.role === 'user' ? new HumanMessage(m.content) : new AIMessage(m.content)
                        );
                        session._documentId = doc.id;
                    }
                } catch (err) {
                    chatLog.warn({ err: err.message }, 'Failed to restore chat history');
                }
            }

            // SSE headers — disable any proxy/middleware buffering
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache, no-transform');
            res.setHeader('Connection', 'keep-alive');
            res.setHeader('X-Accel-Buffering', 'no');
            res.setHeader('Content-Encoding', 'none');
            res.flushHeaders();

            let fullAnswer = '';
            let tokenCount = 0;

            for await (const token of routeQueryStream(session, question, session.chatHistory, req.user?.name)) {
                fullAnswer += token;
                tokenCount++;
                res.write(`data: ${JSON.stringify({ token })}\n\n`);
                // Flush each token immediately for real-time streaming
                if (typeof res.flush === 'function') res.flush();
            }

            res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
            res.end();

            // Update chat history + persist
            session.chatHistory.push(new HumanMessage(question));
            session.chatHistory.push(new AIMessage(fullAnswer));

            if (session.chatHistory.length > 20) {
                session.chatHistory = session.chatHistory.slice(session.chatHistory.length - 20);
            }

            if (req.user && session._documentId) {
                saveChatMessage({ userId: req.user.id, documentId: session._documentId, role: 'user', content: question });
                saveChatMessage({ userId: req.user.id, documentId: session._documentId, role: 'ai', content: fullAnswer });
            }

            const totalMs = Math.round(performance.now() - routeStart);
            chatLog.info(
                { sessionId, totalMs, tokenChunks: tokenCount, answerLength: fullAnswer.length, question: question.substring(0, 60) },
                'Chat stream complete'
            );

        } catch (error) {
            const totalMs = Math.round(performance.now() - routeStart);
            chatLog.error({ err: error.message, totalMs }, 'Chat stream error');
            if (!res.headersSent) {
                res.status(500).json({ error: error.message || "Failed to get answer" });
            } else {
                res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
                res.end();
            }
        }
    });

    // ── Get chat history for a session ──────────────────────────────
    router.get('/chat/history/:sessionId', optionalAuth, async (req, res) => {
        try {
            if (!req.user) return res.status(401).json({ error: 'Authentication required' });

            const doc = await getDocumentBySessionId(req.params.sessionId);
            if (!doc) return res.status(404).json({ error: 'Document not found' });

            const history = await getChatHistory(doc.id, 50);
            res.json({ history });
        } catch (error) {
            chatLog.error({ err: error.message }, 'Chat history error');
            res.status(500).json({ error: 'Failed to load chat history' });
        }
    });

    return router;
}
