import { Router } from 'express';
import { generateQuiz } from "../lib/interview/quizGenerator.js";
import { optionalAuth } from "../middleware/auth.js";
import { saveQuizResult, getDocumentBySessionId, logActivity } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { validate, quizSchema } from '../lib/validation.js';
import { quizLog } from '../lib/logger.js';

export function createQuizRoutes({ sessionCache }) {
    const router = Router();

    router.post('/quiz', validate(quizSchema), optionalAuth, async (req, res) => {
        const routeStart = performance.now();

        try {
            const { sessionId, topic, numQuestions } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
            }

            quizLog.info({ sessionId, topic: topic || 'general', numQuestions: parseInt(numQuestions) || 5 }, 'Quiz request received');

            const quizStart = performance.now();
            const quizData = await generateQuiz(session.vectorStore, {
                topic: topic || "general",
                numQuestions: parseInt(numQuestions) || 5
            });
            const quizMs = Math.round(performance.now() - quizStart);

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

            const totalMs = Math.round(performance.now() - routeStart);
            quizLog.info(
                { sessionId, totalMs, quizMs, questionsGenerated: quizData.quiz?.length || 0, topic: topic || 'general' },
                'Quiz response sent'
            );

            res.json(quizData);

        } catch (error) {
            const totalMs = Math.round(performance.now() - routeStart);
            quizLog.error({ err: error.message, totalMs }, 'Quiz generation error');
            res.status(500).json({ error: error.message || "Failed to generate quiz" });
        }
    });

    return router;
}
