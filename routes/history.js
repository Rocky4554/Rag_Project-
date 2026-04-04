import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import { getUserDocuments, getInterviewResults, getQuizResults, getRecentActivity, deleteDocument, getDocumentBySessionId } from '../lib/db.js';
import { serverLog } from '../lib/logger.js';
import { cleanupSessionAgents } from './voiceAgent.js';

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

export function createHistoryRoutes({ sessionCache, activeAgents, activeVoiceAgents } = {}) {
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

    // ── Delete a document and its related data ──────────────────────
    router.delete('/history/documents/:sessionId', requireAuth, async (req, res) => {
        try {
            const doc = await getDocumentBySessionId(req.params.sessionId);
            if (!doc) return res.status(404).json({ error: 'Document not found' });
            if (doc.user_id !== req.user.id) return res.status(403).json({ error: 'Forbidden' });

            await deleteDocument(doc.id, req.user.id);

            const sessionId = req.params.sessionId;

            // Stop any active agents on this session
            if (activeAgents && activeVoiceAgents) {
                await cleanupSessionAgents(sessionId, { activeAgents, activeVoiceAgents }).catch(() => {});
            }

            // Clear from in-memory session cache
            if (sessionCache && sessionCache[sessionId]) {
                delete sessionCache[sessionId];
                serverLog.info({ sessionId }, 'Session cleared from cache');
            }

            // Try to delete Qdrant collections (text + image, best-effort)
            if (process.env.QDRANT_URL) {
                const headers = process.env.QDRANT_API_KEY ? { 'api-key': process.env.QDRANT_API_KEY } : {};
                const collections = [doc.qdrant_collection].filter(Boolean);
                // Also try the -img variant for image uploads
                if (doc.qdrant_collection && !doc.qdrant_collection.endsWith('-img')) {
                    collections.push(doc.qdrant_collection + '-img');
                }
                for (const col of collections) {
                    try {
                        const resp = await fetch(`${process.env.QDRANT_URL}/collections/${col}`, {
                            method: 'DELETE',
                            headers,
                        });
                        if (resp.ok || resp.status === 404) {
                            serverLog.info({ collection: col, status: resp.status }, 'Qdrant collection deleted');
                        }
                    } catch (qdrantErr) {
                        serverLog.warn({ err: qdrantErr.message, collection: col }, 'Qdrant cleanup failed (non-critical)');
                    }
                }
            }

            res.json({ success: true });
        } catch (error) {
            serverLog.error({ err: error.message, sessionId: req.params.sessionId }, 'Delete document error');
            res.status(500).json({ error: 'Failed to delete document' });
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
