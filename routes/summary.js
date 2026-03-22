import { Router } from 'express';
import { summarizeDocs } from "../lib/interview/summarizer.js";
import { textToAudio } from "../lib/tts/speechToAudio.js";
import { optionalAuth } from "../middleware/auth.js";
import { logActivity } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { validate, summarySchema } from '../lib/validation.js';
import { chatLog as summaryLog } from '../lib/logger.js';

export function createSummaryRoutes({ sessionCache }) {
    const router = Router();

    router.post('/summary', validate(summarySchema), optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session) {
                return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
            }

            summaryLog.info({ sessionId }, 'Generating summary');

            // Use original docs if available, otherwise retrieve from vector store
            let docs = session.docs;
            if (!docs || docs.length === 0) {
                if (session.vectorStore) {
                    summaryLog.info({ sessionId }, 'Retrieving docs from vector store for summary');
                    docs = await session.vectorStore.similaritySearch("summarize the entire document", 20);
                }
                if (!docs || docs.length === 0) {
                    return res.status(404).json({ error: 'No document content available. Please upload the PDF again.' });
                }
            }

            const summary = await summarizeDocs(docs);

            summaryLog.info({ sessionId }, 'Converting summary to audio');
            const audio = await textToAudio(summary);

            // Log activity if user is authenticated
            if (req.user) {
                logActivity({
                    userId: req.user.id,
                    action: 'summary_generated',
                    metadata: { sessionId }
                });
            }

            res.json({ summary, audio });

        } catch (error) {
            summaryLog.error({ err: error.message }, 'Summary/TTS error');
            res.status(500).json({ error: error.message || "Failed to generate summary or audio" });
        }
    });

    return router;
}
