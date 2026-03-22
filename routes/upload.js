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
        try {
            if (!req.file) {
                return res.status(400).json({ error: 'No PDF file uploaded' });
            }

            uploadLog.info({ filename: req.file.filename, size: req.file.size }, 'Processing upload');
            const sessionId = req.file.filename;

            const text = await extractTextFromPDF(req.file.path);
            const docs = await splitText(text);
            const vectorStore = await storeDocuments(docs, sessionId);

            sessionCache[sessionId] = {
                vectorStore,
                docs,
                chatHistory: [],
                interviewState: null,
                createdAt: Date.now()
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

            res.json({
                message: 'PDF processed successfully',
                sessionId
            });

        } catch (error) {
            uploadLog.error({ err: error.message }, 'Upload error');
            res.status(500).json({ error: error.message || "Failed to process PDF" });
        }
    });

    return router;
}
