import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import { getUserDocuments, getInterviewResults, getQuizResults, getRecentActivity } from '../lib/db.js';
import { serverLog } from '../lib/logger.js';

export function createHistoryRoutes() {
    const router = Router();

    // ── Get all user documents ──────────────────────────────────────
    router.get('/history/documents', requireAuth, async (req, res) => {
        try {
            const documents = await getUserDocuments(req.user.id);
            res.json({ documents });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History documents error');
            res.status(500).json({ error: 'Failed to load documents' });
        }
    });

    // ── Get interview results (optionally filtered by document) ─────
    router.get('/history/interviews', requireAuth, async (req, res) => {
        try {
            const results = await getInterviewResults(req.user.id, req.query.documentId);
            res.json({ results });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History interview results error');
            res.status(500).json({ error: 'Failed to load interview results' });
        }
    });

    // ── Get quiz results (optionally filtered by document) ──────────
    router.get('/history/quizzes', requireAuth, async (req, res) => {
        try {
            const results = await getQuizResults(req.user.id, req.query.documentId);
            res.json({ results });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History quiz results error');
            res.status(500).json({ error: 'Failed to load quiz results' });
        }
    });

    // ── Get recent activity ─────────────────────────────────────────
    router.get('/history/activity', requireAuth, async (req, res) => {
        try {
            const limit = parseInt(req.query.limit) || 20;
            const activity = await getRecentActivity(req.user.id, limit);
            res.json({ activity });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History activity error');
            res.status(500).json({ error: 'Failed to load activity' });
        }
    });

    return router;
}
