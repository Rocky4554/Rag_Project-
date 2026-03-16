import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import { AccessToken } from 'livekit-server-sdk';

import { extractTextFromPDF } from "./lib/pdfLoader.js";
import { splitText } from "./lib/textSplitter.js";
import { storeDocuments } from "./lib/vectorStore.js";
import { generateQuiz } from "./lib/quizGenerator.js";
import { summarizeDocs } from "./lib/summarizer.js";
import { textToAudio } from "./lib/speechToAudio.js";
import { queryRAG } from "./lib/rag.js";
import { createInterviewAgent, registerVectorStore, registerSocket, unregisterSocket, parseTTSResponse, checkTTSCache } from "./lib/interviewAgent.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { InterviewAgentWorker } from "./agent/agent.js";

// Global map to hold active agents
const activeAgents = new Map();
const require = createRequire(import.meta.url);
const { DeepgramClient } = require('@deepgram/sdk');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new SocketIOServer(httpServer, {
    cors: { origin: "*", methods: ["GET", "POST"] }
});
const PORT = process.env.PORT || 3000;

console.log(`\n========================================`);
console.log(`🔑 API Key Verification:`);
console.log(`   GROQ_API_KEY      : ${process.env.GROQ_API_KEY ? '✅ Loaded' : '❌ Missing'}`);
console.log(`   GEMINI_API_KEY    : ${process.env.GEMINI_API_KEY ? '✅ Loaded' : '❌ Missing/Not Used'}`);
console.log(`   AWS_ACCESS_KEY_ID : ${process.env.AWS_ACCESS_KEY_ID ? '✅ Loaded' : '❌ Missing'}`);
console.log(`   KOKORO_API_URL    : ${process.env.KOKORO_API_URL ? '✅ Loaded' : '⚠️ Defaulting'}`);
console.log(`   LIVEKIT_API_KEY   : ${process.env.LIVEKIT_API_KEY ? '✅ Loaded' : '❌ Missing'}`);
console.log(`   QDRANT_URL        : ${process.env.QDRANT_URL ? '✅ Loaded' : '❌ Missing'}`);
console.log(`   QDRANT_API_KEY    : ${process.env.QDRANT_API_KEY ? '✅ Loaded' : '❌ Missing'}`);
console.log(`========================================\n`);

// ── Socket.io: register socket per session for AI streaming ──────
io.on('connection', (socket) => {
    console.log(`[Socket.io] Client connected: ${socket.id}`);

    socket.on('register_session', (sessionId) => {
        if (sessionId) {
            socket.join(sessionId);
            socket.data.sessionId = sessionId;
            registerSocket(sessionId, socket);
            console.log(`[Socket.io] Socket ${socket.id} registered for session: ${sessionId}`);
        }
    });

    socket.on('disconnect', () => {
        if (socket.data.sessionId) {
            unregisterSocket(socket.data.sessionId);
        }
        console.log(`[Socket.io] Client disconnected: ${socket.id}`);
    });
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Set up multer for PDF uploads
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
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

const sessionCache = {};

// Routes
app.post('/api/upload', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No PDF file uploaded' });
        }

        console.log(`Received file: ${req.file.filename}`);
        const sessionId = req.file.filename;

        const text = await extractTextFromPDF(req.file.path);
        const docs = await splitText(text);
        const vectorStore = await storeDocuments(docs, sessionId);

        sessionCache[sessionId] = {
            vectorStore: vectorStore,
            docs: docs,
            chatHistory: [],
            interviewState: null
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

        const answer = await queryRAG(session.vectorStore, question, session.chatHistory);

        session.chatHistory.push(new HumanMessage(question));
        session.chatHistory.push(new AIMessage(answer));

        if (session.chatHistory.length > 20) {
            session.chatHistory = session.chatHistory.slice(session.chatHistory.length - 20);
        }

        res.json({ answer: answer });

    } catch (error) {
        console.error("Chat error:", error);
        res.status(500).json({ error: error.message || "Failed to get answer" });
    }
});

// ==========================================
// LiveKit — Token Endpoint
// ==========================================
app.post('/api/livekit/token', async (req, res) => {
    try {
        const { sessionId } = req.body;
        if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

        const apiKey    = process.env.LIVEKIT_API_KEY;
        const apiSecret = process.env.LIVEKIT_API_SECRET;
        const livekitUrl = process.env.LIVEKIT_URL;

        if (!apiKey || !apiSecret || !livekitUrl) {
            return res.status(500).json({ error: 'LiveKit environment variables not configured.' });
        }

        // Each sessionId maps to a LiveKit room — one room per interview
        const at = new AccessToken(apiKey, apiSecret, {
            identity: `candidate-${sessionId}`,
            ttl: '2h',
        });

        at.addGrant({
            roomJoin:     true,
            room:         sessionId,   // room = sessionId (unique per PDF upload)
            canPublish:   true,        // candidate can send mic audio
            canSubscribe: true,        // candidate can receive AI audio
        });

        const token = await at.toJwt();
        console.log(`[LiveKit] Token generated for session: ${sessionId}`);

        res.json({ token, url: livekitUrl });

    } catch (error) {
        console.error('[LiveKit] Token generation error:', error);
        res.status(500).json({ error: error.message || 'Failed to generate LiveKit token' });
    }
});

// ==========================================
// Deepgram — Token Endpoint
// ==========================================
app.get('/api/deepgram/token', async (req, res) => {
    try {
        const apiKey = process.env.DEEPGRAM_API_KEY;
        if (!apiKey) {
            return res.status(500).json({ error: 'DEEPGRAM_API_KEY not configured' });
        }

        // STT runtime options (configurable from .env)
        const model = process.env.DEEPGRAM_STT_MODEL || 'nova-3';
        const language = process.env.DEEPGRAM_STT_LANGUAGE || 'en';
        const smartFormat = (process.env.DEEPGRAM_STT_SMART_FORMAT || 'true').toLowerCase() === 'true';
        const interimResults = (process.env.DEEPGRAM_STT_INTERIM_RESULTS || 'true').toLowerCase() === 'true';
        const endpointing = parseInt(process.env.DEEPGRAM_STT_ENDPOINTING_MS || '500', 10);

        // Simplified: returns API key as token for browser websocket auth.
        // For stricter production security, replace with short-lived project keys/proxy.
        res.json({
            token: apiKey,
            stt: {
                model,
                language,
                smart_format: smartFormat,
                interim_results: interimResults,
                endpointing: Number.isFinite(endpointing) ? endpointing : 500,
                encoding: 'linear16',
                sample_rate: 16000,
                channels: 1
            }
        });
    } catch (error) {
        console.error('[Deepgram] Token generation error:', error);
        res.status(500).json({ error: 'Failed to retrieve Deepgram token' });
    }
});

// ==========================================
// AI Voice Interview Routes
// ==========================================

const interviewAgent = createInterviewAgent();

app.post('/api/interview/start', async (req, res) => {
    try {
        const { sessionId, maxQuestions = 5 } = req.body;
        if (!sessionId) return res.status(400).json({ error: 'Session ID is required' });

        const session = sessionCache[sessionId];
        if (!session || !session.vectorStore) {
            return res.status(404).json({ error: 'Session not found. Please upload the PDF again.' });
        }

        console.log(`Starting interview for session: ${sessionId}`);

        registerVectorStore(sessionId, session.vectorStore);

        const initialState = {
            sessionId: sessionId,
            maxQuestions: parseInt(maxQuestions),
            difficultyLevel: "medium",
            chatHistory: [],
            questionsAsked: 0,
            topicsUsed: []
        };

        const config = { configurable: { thread_id: sessionId } };
        const resultState = await interviewAgent.invoke(initialState, config);

        session.interviewStateConfig = config;

        // Start the Agent Worker in the background
        if (activeAgents.has(sessionId)) {
            const oldAgent = activeAgents.get(sessionId);
            oldAgent.stop();
        }
        
        const agent = new InterviewAgentWorker(sessionId, sessionCache, interviewAgent, io);
        activeAgents.set(sessionId, agent);
        
        // Let agent connect to LiveKit, then manually trigger the first question
        agent.start().then(() => {
            const { uniquePart } = parseTTSResponse(resultState.currentQuestion);
            // We tell the UI the question, and tell the agent to speak it
            io.to(sessionId).emit('transcript_final', { role: 'ai', text: uniquePart });
            agent.speak("Hello! Welcome to your AI voice interview. " + uniquePart);
        }).catch(err => {
            console.error("[Agent Start Error]", err);
        });

        // Simplified response, audio is handled via LiveKit now
        res.json({
            questionNumber: resultState.questionsAsked,
            difficulty: resultState.difficultyLevel,
            agentStarted: true
        });

    } catch (error) {
        console.error("Interview start error:", error);
        res.status(500).json({ error: error.message || "Failed to start interview" });
    }
});

// No longer needed, agent handles this internally via SessionBridge

// Start the server
httpServer.listen(PORT, () => {
    console.log(`\n========================================`);
    console.log(`🚀 Quiz Server running on http://localhost:${PORT}`);
    console.log(`🔌 Socket.io streaming enabled`);
    console.log(`========================================\n`);
});