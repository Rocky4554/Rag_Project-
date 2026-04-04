import { Router } from 'express';
import os from 'os';
import path from 'path';
import fs from 'fs/promises';
import { randomBytes } from 'crypto';
import { QdrantClient } from "@qdrant/js-client-rest";
import { extractTextFromPDF } from "../lib/pipeline/pdfLoader.js";
import { splitText } from "../lib/pipeline/textSplitter.js";
import { storeDocuments } from "../lib/pipeline/vectorStore.js";
import { imageEmbedder } from "../lib/embeddings.js";
import { optionalAuth } from "../middleware/auth.js";
import { saveDocument, logActivity } from "../lib/db.js";
import { uploadLog } from '../lib/logger.js';
import { Document } from "@langchain/core/documents";

// Qdrant client for direct vector operations (used for image embeddings
// which have a different dimensionality than text embeddings)
const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});

const IMAGE_EMBEDDING_DIM = 1024; // Jina CLIP v2 output dimension

/**
 * Ensure a Qdrant collection exists for image vectors.
 * Creates it with cosine distance if it doesn't already exist.
 */
async function ensureImageCollection(collectionName) {
    try {
        await qdrantClient.getCollection(collectionName);
    } catch {
        // Collection doesn't exist — create it
        await qdrantClient.createCollection(collectionName, {
            vectors: {
                size: IMAGE_EMBEDDING_DIM,
                distance: "Cosine",
            },
        });
        uploadLog.info({ collection: collectionName, dim: IMAGE_EMBEDDING_DIM }, 'Created Qdrant image collection');
    }
}

export function createUploadRoutes({ sessionCache, upload }) {
    const router = Router();

    // ── PDF & Image file upload ──────────────────────────────────────
    router.post('/upload', upload.single('file'), optionalAuth, async (req, res) => {
        const totalStart = performance.now();

        try {
            if (!req.file) {
                return res.status(400).json({ error: 'No file uploaded' });
            }

            const mimetype = req.file.mimetype;
            const isPdf = mimetype === 'application/pdf';
            const isImage = mimetype.startsWith('image/');

            if (!isPdf && !isImage) {
                return res.status(400).json({ error: 'Unsupported file type' });
            }

            uploadLog.info(
                { originalName: req.file.originalname, size: req.file.size, type: isPdf ? 'pdf' : 'image' },
                'Upload processing started'
            );

            const sessionId = randomBytes(12).toString('hex') + '-' + Date.now();

            if (isPdf) {
                // ── PDF flow (existing) ─────────────────────────────
                const tmpPath = path.join(os.tmpdir(), `rag-${sessionId}.pdf`);
                await fs.writeFile(tmpPath, req.file.buffer);

                // Step 1: PDF extraction
                const extractStart = performance.now();
                let text;
                try {
                    text = await extractTextFromPDF(tmpPath);
                } finally {
                    fs.unlink(tmpPath).catch(() => {});
                }
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
                    contentType: 'pdf',
                    hasImages: false,
                };

                // Persist to Supabase if authenticated
                if (req.user) {
                    try {
                        const doc = await saveDocument({
                            userId: req.user.id,
                            sessionId,
                            filename: req.file.originalname,
                            originalName: req.file.originalname,
                            qdrantCollection: sessionId,
                            chunkCount: docs.length,
                        });
                        logActivity({
                            userId: req.user.id,
                            action: 'document_uploaded',
                            metadata: { sessionId, originalName: req.file.originalname, chunkCount: docs.length },
                        });
                        uploadLog.info({ docId: doc.id, sessionId }, 'Document saved to Supabase');
                    } catch (dbErr) {
                        uploadLog.warn({ err: dbErr.message }, 'DB save failed (continuing)');
                    }
                }

                const totalMs = Math.round(performance.now() - totalStart);
                uploadLog.info(
                    { sessionId, totalMs, extractMs, splitMs, storeMs, chunks: docs.length, filename: req.file.originalname },
                    'Upload pipeline complete (PDF)'
                );

                return res.json({ message: 'PDF processed successfully', sessionId });

            } else if (isImage) {
                // ── Image flow (new) ────────────────────────────────
                if (!imageEmbedder) {
                    return res.status(503).json({
                        error: 'Image embedding is not available. JINA_API_KEY is required for image uploads.',
                    });
                }

                // Step 1: Convert image buffer to base64
                const base64 = req.file.buffer.toString('base64');
                uploadLog.info({ step: 'base64', size: base64.length }, 'Image converted to base64');

                // Step 2: Embed image with Jina CLIP v2
                const embedStart = performance.now();
                const imageVector = await imageEmbedder.embedImage(base64);
                const embedMs = Math.round(performance.now() - embedStart);
                uploadLog.info({ step: 'embed', durationMs: embedMs, dim: imageVector.length }, 'Image embedded with Jina CLIP');

                // Step 3: Store image vector in Qdrant (separate collection due to different dims)
                const imageCollection = `${sessionId}-img`;
                const storeStart = performance.now();
                await ensureImageCollection(imageCollection);

                const pointId = randomBytes(16).toString('hex');
                // Store base64 in payload so it can be retrieved later;
                // also store filename and mimetype for context.
                await qdrantClient.upsert(imageCollection, {
                    wait: true,
                    points: [
                        {
                            id: pointId,
                            vector: imageVector,
                            payload: {
                                filename: req.file.originalname,
                                mimetype: req.file.mimetype,
                                base64,
                                uploadedAt: new Date().toISOString(),
                            },
                        },
                    ],
                });
                const storeMs = Math.round(performance.now() - storeStart);
                uploadLog.info({ step: 'store', durationMs: storeMs, collection: imageCollection }, 'Image vector stored in Qdrant');

                sessionCache[sessionId] = {
                    vectorStore: null, // no text vector store for image-only uploads
                    docs: [],
                    chatHistory: [],
                    interviewState: null,
                    createdAt: Date.now(),
                    originalName: req.file.originalname,
                    contentType: 'image',
                    hasImages: true,
                    imageCollection,
                };

                // Persist to Supabase if authenticated
                if (req.user) {
                    try {
                        const doc = await saveDocument({
                            userId: req.user.id,
                            sessionId,
                            filename: req.file.originalname,
                            originalName: req.file.originalname,
                            qdrantCollection: imageCollection,
                            chunkCount: 1,
                        });
                        logActivity({
                            userId: req.user.id,
                            action: 'image_uploaded',
                            metadata: { sessionId, originalName: req.file.originalname },
                        });
                        uploadLog.info({ docId: doc.id, sessionId }, 'Image document saved to Supabase');
                    } catch (dbErr) {
                        uploadLog.warn({ err: dbErr.message }, 'DB save failed (continuing)');
                    }
                }

                const totalMs = Math.round(performance.now() - totalStart);
                uploadLog.info(
                    { sessionId, totalMs, embedMs, storeMs, filename: req.file.originalname },
                    'Upload pipeline complete (image)'
                );

                return res.json({ message: 'Image processed successfully', sessionId });
            }
        } catch (error) {
            const totalMs = Math.round(performance.now() - totalStart);
            uploadLog.error({ err: error.message, totalMs }, 'Upload error');
            res.status(500).json({ error: error.message || "Failed to process upload" });
        }
    });

    // ── Pasted text upload ───────────────────────────────────────────
    router.post('/upload/text', optionalAuth, async (req, res) => {
        const totalStart = performance.now();

        try {
            const { text, title } = req.body;

            if (!text || typeof text !== 'string' || text.trim().length === 0) {
                return res.status(400).json({ error: 'No text provided. Send { "text": "...", "title": "..." } in the request body.' });
            }

            const trimmedText = text.trim();
            const docTitle = (title && typeof title === 'string' ? title.trim() : 'Pasted Text') || 'Pasted Text';

            uploadLog.info({ title: docTitle, textLength: trimmedText.length }, 'Text upload processing started');

            const sessionId = randomBytes(12).toString('hex') + '-' + Date.now();

            // Step 1: Wrap raw text in a LangChain Document for the splitter
            const rawDocs = [
                new Document({
                    pageContent: trimmedText,
                    metadata: { source: docTitle, type: 'pasted_text' },
                }),
            ];

            // Step 2: Text splitting
            const splitStart = performance.now();
            const docs = await splitText(rawDocs);
            const splitMs = Math.round(performance.now() - splitStart);
            uploadLog.info({ step: 'split', durationMs: splitMs, chunks: docs.length }, 'Text split into chunks');

            // Step 3: Embedding + vector store
            const storeStart = performance.now();
            const vectorStore = await storeDocuments(docs, sessionId);
            const storeMs = Math.round(performance.now() - storeStart);
            uploadLog.info({ step: 'embed+store', durationMs: storeMs, chunks: docs.length }, 'Text embedded and stored in Qdrant');

            sessionCache[sessionId] = {
                vectorStore,
                docs,
                chatHistory: [],
                interviewState: null,
                createdAt: Date.now(),
                originalName: docTitle,
                contentType: 'text',
                hasImages: false,
            };

            // Persist to Supabase if authenticated
            if (req.user) {
                try {
                    const doc = await saveDocument({
                        userId: req.user.id,
                        sessionId,
                        filename: docTitle,
                        originalName: docTitle,
                        qdrantCollection: sessionId,
                        chunkCount: docs.length,
                    });
                    logActivity({
                        userId: req.user.id,
                        action: 'text_uploaded',
                        metadata: { sessionId, title: docTitle, chunkCount: docs.length },
                    });
                    uploadLog.info({ docId: doc.id, sessionId }, 'Text document saved to Supabase');
                } catch (dbErr) {
                    uploadLog.warn({ err: dbErr.message }, 'DB save failed (continuing)');
                }
            }

            const totalMs = Math.round(performance.now() - totalStart);
            uploadLog.info(
                { sessionId, totalMs, splitMs, storeMs, chunks: docs.length, title: docTitle },
                'Upload pipeline complete (text)'
            );

            res.json({ message: 'Text processed successfully', sessionId });

        } catch (error) {
            const totalMs = Math.round(performance.now() - totalStart);
            uploadLog.error({ err: error.message, totalMs }, 'Text upload error');
            res.status(500).json({ error: error.message || "Failed to process text" });
        }
    });

    return router;
}
