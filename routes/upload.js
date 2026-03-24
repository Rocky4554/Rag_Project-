import { Router } from 'express';
import { extractTextFromPDF } from "../lib/pipeline/pdfLoader.js";
import { splitText } from "../lib/pipeline/textSplitter.js";
import { storeDocuments } from "../lib/pipeline/vectorStore.js";
import { optionalAuth } from "../middleware/auth.js";
import { saveDocument, logActivity } from "../lib/db.js";
import { uploadLog } from '../lib/logger.js';

export function createUploadRoutes({ sessionCache, upload }) {
    const router = Router();

    router.post('/upload', upload.single('pdf'), optionalAuth, async (req, res) => {
        const totalStart = performance.now();

        try {
            if (!req.file) {
                return res.status(400).json({ error: 'No PDF file uploaded' });
            }

            uploadLog.info({ filename: req.file.filename, originalName: req.file.originalname, size: req.file.size }, 'Upload processing started');
            const sessionId = req.file.filename;

            // Step 1: PDF extraction
            const extractStart = performance.now();
            const text = await extractTextFromPDF(req.file.path);
            const extractMs = Math.round(performance.now() - extractStart);
            uploadLog.info({ step: 'extract', durationMs: extractMs, pages: text.length }, 'PDF text extracted');

            // Step 2: Text splitting
            const splitStart = performance.now();
            const docs = await splitText(text);
            const splitMs = Math.round(performance.now() - splitStart);
            uploadLog.info({ step: 'split', durationMs: splitMs, chunks: docs.length }, 'Text split into chunks');

            // Step 3: Embedding + vector store
            const storeStart = performance.now();
            const vectorStore = await storeDocuments(docs, sessionId);
            const storeMs = Math.round(performance.now() - storeStart);
            uploadLog.info({ step: 'embed+store', durationMs: storeMs, chunks: docs.length }, 'Documents embedded and stored in Qdrant');

            sessionCache[sessionId] = {
                vectorStore,
                docs,
                chatHistory: [],
                interviewState: null,
                createdAt: Date.now(),
                originalName: req.file.originalname,
            };

            // Persist document metadata to Supabase if user is authenticated
            if (req.user) {
                try {
                    const doc = await saveDocument({
                        userId: req.user.id,
                        sessionId,
                        filename: req.file.filename,
                        originalName: req.file.originalname,
                        qdrantCollection: sessionId,
                        chunkCount: docs.length
                    });
                    logActivity({
                        userId: req.user.id,
                        action: 'document_uploaded',
                        metadata: { sessionId, originalName: req.file.originalname, chunkCount: docs.length }
                    });
                    uploadLog.info({ docId: doc.id, sessionId }, 'Document saved to Supabase');
                } catch (dbErr) {
                    uploadLog.warn({ err: dbErr.message }, 'DB save failed (continuing)');
                }
            }

            const totalMs = Math.round(performance.now() - totalStart);
            uploadLog.info(
                { sessionId, totalMs, extractMs, splitMs, storeMs, chunks: docs.length, filename: req.file.originalname },
                'Upload pipeline complete'
            );

            res.json({
                message: 'PDF processed successfully',
                sessionId
            });

        } catch (error) {
            const totalMs = Math.round(performance.now() - totalStart);
            uploadLog.error({ err: error.message, totalMs }, 'Upload error');
            res.status(500).json({ error: error.message || "Failed to process PDF" });
        }
    });

    return router;
}
