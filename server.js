import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

import { extractTextFromPDF } from "./lib/pdfLoader.js";
import { splitText } from "./lib/textSplitter.js";
import { storeDocuments } from "./lib/vectorStore.js";
import { generateQuiz } from "./lib/quizGenerator.js";
import { summarizeDocs, textToAudio } from "./lib/summarizer.js";
import { queryRAG } from "./lib/rag.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Set up multer for PDF uploads
// Create uploads dir if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        // Keep original extension, add timestamp to avoid collisions
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname))
    }
});

const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/pdf') {
            cb(null, true);
        } else {
            cb(new Error('Only PDF files are allowed!'), false);
        }
    }
});

// In-memory store mapping session IDs to vector stores
// In a real app, you'd probably use a real DB or re-initialize vector stores from ChromaDB by collection name
const sessionCache = {};

// Routes
app.post('/api/upload', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No PDF file uploaded' });
        }

        console.log(`Received file: ${req.file.filename}`);
        const sessionId = req.file.filename;

        // Process the PDF
        const text = await extractTextFromPDF(req.file.path);
        const docs = await splitText(text);

        // Use the filename as a unique collection name for this user's PDF
        const vectorStore = await storeDocuments(docs, sessionId);

        // Cache the vector store, raw docs, and an empty chat history array
        sessionCache[sessionId] = {
            vectorStore: vectorStore,
            docs: docs,
            chatHistory: []
        };

        res.json({
            message: 'PDF processed successfully',
            sessionId: sessionId
        });

    } catch (error) {
        console.error("Upload error:", error);
        res.status(500).json({ error: error.message || "Failed to process PDF" });
    }
});

app.post('/api/quiz', async (req, res) => {
    try {
        const { sessionId, topic, numQuestions } = req.body;

        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }

        const session = sessionCache[sessionId];
        if (!session || !session.vectorStore) {
            return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
        }
        const vectorStore = session.vectorStore;

        // Generate the quiz
        const quizData = await generateQuiz(vectorStore, {
            topic: topic || "general",
            numQuestions: parseInt(numQuestions) || 5
        });

        res.json(quizData);

    } catch (error) {
        console.error("Quiz generation error:", error);
        res.status(500).json({ error: error.message || "Failed to generate quiz" });
    }
});

app.post('/api/summary', async (req, res) => {
    try {
        const { sessionId } = req.body;

        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }

        const session = sessionCache[sessionId];
        if (!session || !session.docs) {
            return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
        }

        console.log(`Generating summary for session: ${sessionId}`);
        const summary = await summarizeDocs(session.docs);

        console.log(`Converting summary to audio...`);
        const audio = await textToAudio(summary);

        res.json({
            summary: summary,
            audio: audio
        });

    } catch (error) {
        console.error("Summary/TTS error:", error);
        res.status(500).json({ error: error.message || "Failed to generate summary or audio" });
    }
});

app.post('/api/chat', async (req, res) => {
    try {
        const { sessionId, question } = req.body;

        if (!sessionId || !question) {
            return res.status(400).json({ error: 'Session ID and question are required' });
        }

        const session = sessionCache[sessionId];
        if (!session || !session.vectorStore) {
            return res.status(404).json({ error: 'Session not found or expired. Please upload the PDF again.' });
        }

        console.log(`Chat request: "${question}" for session: ${sessionId}`);

        // Use the stored history to understand follow-up questions
        const answer = await queryRAG(session.vectorStore, question, session.chatHistory);

        // Save this exchange to the session's memory for the next follow-up question
        session.chatHistory.push(new HumanMessage(question));
        session.chatHistory.push(new AIMessage(answer));

        // Limit history to last 10 exchanges (20 messages) to prevent context bloat
        if (session.chatHistory.length > 20) {
            session.chatHistory = session.chatHistory.slice(session.chatHistory.length - 20);
        }

        res.json({ answer: answer });

    } catch (error) {
        console.error("Chat error:", error);
        res.status(500).json({ error: error.message || "Failed to get answer" });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`\n========================================`);
    console.log(`🚀 Quiz Server running on http://localhost:${PORT}`);
    console.log(`========================================\n`);
});
