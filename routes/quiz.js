import { Router } from 'express';
import { generateQuiz } from "../lib/interview/quizGenerator.js";
import { optionalAuth } from "../middleware/auth.js";
import { saveQuizResult, getDocumentBySessionId, logActivity } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { validate, quizSchema } from '../lib/validation.js';
import { chatLog as quizLog } from '../lib/logger.js';

export function createQuizRoutes({ sessionCache }) {
    const router = Router();

    router.post('/quiz', validate(quizSchema), optionalAuth, async (req, res) => {
        try {
            const { sessionId, topic, numQuestions } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
            }

            const quizData = await generateQuiz(session.vectorStore, {
                topic: topic || "general",
                numQuestions: parseInt(numQuestions) || 5
            });

            // Persist quiz to Supabase if user is authenticated
            if (req.user) {
                try {
                    const doc = await getDocumentBySessionId(sessionId);
                    if (doc) {
                        await saveQuizResult({
                            userId: req.user.id,
                            documentId: doc.id,
                            topic: topic || "general",
                            questions: quizData.questions || quizData,
                            score: null,
                            totalQuestions: parseInt(numQuestions) || 5
                        });
                        logActivity({
                            userId: req.user.id,
                            action: 'quiz_generated',
                            metadata: { sessionId, topic: topic || "general", numQuestions: parseInt(numQuestions) || 5 }
                        });
                    }
                } catch (dbErr) {
                    quizLog.warn({ err: dbErr.message }, 'Quiz DB save failed (continuing)');
                }
            }

            res.json(quizData);

        } catch (error) {
            quizLog.error({ err: error.message }, 'Quiz generation error');
            res.status(500).json({ error: error.message || "Failed to generate quiz" });
        }
    });

    return router;
}
