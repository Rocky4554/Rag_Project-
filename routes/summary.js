import { Router } from 'express';
import { summarizeDocs } from "../lib/interview/summarizer.js";
import { textToAudio, audioMimeType } from "../lib/tts/speechToAudio.js";
import { optionalAuth } from "../middleware/auth.js";
import { logActivity } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { validate, summarySchema } from '../lib/validation.js';
import { summaryLog } from '../lib/logger.js';

export function createSummaryRoutes({ sessionCache }) {
    const router = Router();

    router.post('/summary', validate(summarySchema), optionalAuth, async (req, res) => {
        const routeStart = performance.now();

        try {
            const { sessionId } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session) {
                return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
            }

            summaryLog.info({ sessionId }, 'Summary request received');

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

            const llmStart = performance.now();
            const summary = await summarizeDocs(docs);
            const llmMs = Math.round(performance.now() - llmStart);
            summaryLog.info({ sessionId, llmMs, summaryLength: summary.length }, 'Summary text generated');

            const ttsStart = performance.now();
            const audio = await textToAudio(summary);
            const ttsMs = Math.round(performance.now() - ttsStart);
            summaryLog.info({ sessionId, ttsMs, audioSize: audio?.length || 0 }, 'Summary audio generated');

            // Log activity if user is authenticated
            if (req.user) {
                logActivity({
                    userId: req.user.id,
                    action: 'summary_generated',
                    metadata: { sessionId }
                });
            }

            const totalMs = Math.round(performance.now() - routeStart);
            summaryLog.info(
                { sessionId, totalMs, llmMs, ttsMs, summaryWords: summary.split(/\s+/).length },
                'Summary response sent'
            );

            res.json({ summary, audio, mimeType: audioMimeType });

        } catch (error) {
            const totalMs = Math.round(performance.now() - routeStart);
            summaryLog.error({ err: error.message, totalMs }, 'Summary/TTS error');
            res.status(500).json({ error: error.message || "Failed to generate summary or audio" });
        }
    });

    return router;
}
