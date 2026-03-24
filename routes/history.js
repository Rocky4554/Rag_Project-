import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import { getUserDocuments, getInterviewResults, getQuizResults, getRecentActivity } from '../lib/db.js';
import { serverLog } from '../lib/logger.js';

const mapDocument = (d) => ({
    id: d.id,
    sessionId: d.session_id,
    filename: d.filename,
    originalName: d.original_name || d.filename,
    chunkCount: d.chunk_count,
    createdAt: d.created_at,
});

const mapInterview = (r) => ({
    id: r.id,
    sessionId: r.thread_id,
    documentId: r.document_id,
    documentName: r.documents?.original_name || r.documents?.filename,
    questionsAsked: r.questions_asked,
    difficultyLevel: r.difficulty_level,
    finalReport: r.final_report,
    scores: r.scores,
    topicScores: r.topic_scores,
    createdAt: r.created_at,
});

const mapQuiz = (r) => ({
    id: r.id,
    sessionId: r.documents?.session_id || null,
    documentId: r.document_id,
    documentName: r.documents?.original_name || r.documents?.filename,
    topic: r.topic,
    totalQuestions: r.total_questions,
    score: r.score,
    createdAt: r.created_at,
});

const mapActivity = (a) => ({
    id: a.id,
    action: a.action,
    metadata: a.metadata,
    createdAt: a.created_at,
});

export function createHistoryRoutes() {
    const router = Router();

    // ── Get all user documents ──────────────────────────────────────
    router.get('/history/documents', requireAuth, async (req, res) => {
        try {
            const documents = await getUserDocuments(req.user.id);
            res.json({ documents: documents.map(mapDocument) });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History documents error');
            res.status(500).json({ error: 'Failed to load documents' });
        }
    });

    // ── Get interview results (optionally filtered by document) ─────
    router.get('/history/interviews', requireAuth, async (req, res) => {
        try {
            const results = await getInterviewResults(req.user.id, req.query.documentId);
            res.json({ results: results.map(mapInterview) });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History interview results error');
            res.status(500).json({ error: 'Failed to load interview results' });
        }
    });

    // ── Get quiz results (optionally filtered by document) ──────────
    router.get('/history/quizzes', requireAuth, async (req, res) => {
        try {
            const results = await getQuizResults(req.user.id, req.query.documentId);
            res.json({ results: results.map(mapQuiz) });
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
            res.json({ activity: activity.map(mapActivity) });
        } catch (error) {
            serverLog.error({ err: error.message }, 'History activity error');
            res.status(500).json({ error: 'Failed to load activity' });
        }
    });

    return router;
}
