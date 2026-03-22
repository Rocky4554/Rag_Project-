import { Router } from 'express';
import { optionalAuth, requireAuth } from '../middleware/auth.js';
import { ensureSession } from '../lib/sessionRestore.js';
import { getUserDocuments } from '../lib/db.js';
import { serverLog } from '../lib/logger.js';

export function createSessionRoutes({ sessionCache }) {
    const router = Router();

    /**
     * POST /api/session/restore
     * Attempts to restore a session from DB+Qdrant into memory.
     * Returns session status (active, restored, or not_found).
     */
    router.post('/session/restore', optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) {
                return res.status(400).json({ error: 'sessionId is required' });
            }

            // Check if already in memory
            if (sessionCache[sessionId]?.vectorStore) {
                return res.json({ status: 'active', sessionId });
            }

            // Try to restore
            const session = await ensureSession(sessionCache, sessionId);
            if (session) {
                return res.json({ status: 'restored', sessionId });
            }

            return res.json({ status: 'not_found', sessionId });

        } catch (error) {
            serverLog.error({ err: error.message }, 'Session restore endpoint error');
            res.status(500).json({ error: 'Failed to restore session' });
        }
    });

    /**
     * GET /api/session/status/:sessionId
     * Check if a session is active in memory (without restoring).
     */
    router.get('/session/status/:sessionId', (req, res) => {
        const { sessionId } = req.params;
        const session = sessionCache[sessionId];

        res.json({
            sessionId,
            active: !!(session?.vectorStore),
            hasInterview: !!(session?.interviewStateConfig),
            hasChatHistory: (session?.chatHistory?.length || 0) > 0,
        });
    });

    /**
     * GET /api/session/user-sessions
     * Returns all sessions for the authenticated user that can be restored.
     * This lets the frontend show a "resume session" picker.
     */
    router.get('/session/user-sessions', requireAuth, async (req, res) => {
        try {
            const documents = await getUserDocuments(req.user.id);

            const sessions = documents.map(doc => ({
                sessionId: doc.session_id,
                originalName: doc.original_name,
                createdAt: doc.created_at,
                chunkCount: doc.chunk_count,
                active: !!(sessionCache[doc.session_id]?.vectorStore),
            }));

            res.json({ sessions });

        } catch (error) {
            serverLog.error({ err: error.message }, 'User sessions error');
            res.status(500).json({ error: 'Failed to load sessions' });
        }
    });

    return router;
}
