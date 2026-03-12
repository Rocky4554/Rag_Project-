import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
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
import { createRequire } from 'module';
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
console.log(`========================================\n`);

// ── Socket.io: register socket per session for AI streaming ──────
io.on('connection', (socket) => {
    console.log(`[Socket.io] Client connected: ${socket.id}`);

    socket.on('register_session', (sessionId) => {
        if (sessionId) {
            registerSocket(sessionId, socket);
            console.log(`[Socket.io] Socket ${socket.id} registered for session: ${sessionId}`);
        }
    });

    socket.on('disconnect', () => {
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
        // Simplified: Returning the main API key directly to avoid "createProjectKey is not a function" errors
        // In highly secure production environments, consider a server-side proxy instead of passing keys to the client.
        res.json({ token: apiKey });
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

        console.log(`🔍 FIRST QUESTION GENERATED: "${resultState.currentQuestion}"`);

        // ═══════════════════════════════════════════════════════════════
        // CRITICAL FIX: Parse the question tag properly
        // Agent generates: "[interview_intro] What is a linked list?"
        // We should: play interview_intro.mp3 + TTS("What is a linked list?")
        // ═══════════════════════════════════════════════════════════════
        const { phraseKey, uniquePart } = parseTTSResponse(resultState.currentQuestion);

        console.log(`🔍 PARSED: phraseKey="${phraseKey}", uniquePart="${uniquePart}"`);

        // Build audio response
        let introCachedAudio = null;
        let questionAudio = null;

        // If there's a tag in the question (like interview_intro or next_question)
        if (phraseKey) {
            const cachedFile = checkTTSCache(phraseKey);
            if (cachedFile) {
                console.log(`[TTS] Cache hit for question tag: ${phraseKey}`);
                introCachedAudio = fs.readFileSync(cachedFile).toString('base64');
            }
        }

        // Generate TTS for the unique part of the question (without the tag)
        if (uniquePart && uniquePart.trim().length > 0) {
            console.log(`[TTS] Generating speech for question: "${uniquePart}"`);
            const ttsResult = await textToAudio(uniquePart);
            questionAudio = ttsResult?.audio ?? ttsResult;
        }

        res.json({
            question: uniquePart,
            audio: questionAudio,
            feedbackCachedAudio: introCachedAudio, // Frontend plays this first!
            questionNumber: resultState.questionsAsked,
            difficulty: resultState.difficultyLevel
        });

    } catch (error) {
        console.error("Interview start error:", error);
        res.status(500).json({ error: error.message || "Failed to start interview" });
    }
});

app.post('/api/interview/answer', async (req, res) => {
    try {
        const { sessionId, answer } = req.body;
        if (!sessionId || !answer) return res.status(400).json({ error: 'Session ID and answer are required' });

        const session = sessionCache[sessionId];
        if (!session || !session.interviewStateConfig) {
            return res.status(404).json({ error: 'Interview session not found. Please start the interview first.' });
        }

        console.log(`Processing interview answer for session: ${sessionId}`);
        console.log(`🔍 USER SAID: "${answer}"`);

        const inputState = { userAnswer: answer };
        const resultState = await interviewAgent.invoke(inputState, session.interviewStateConfig);

        let finalReport = null;
        let responsePayload = {
            evaluation: resultState.evaluation || {
                score: 5, accuracy: 5, clarity: 5, depth: 5,
                feedback: "Thank you for your answer.",
                nextDifficulty: "same"
            }
        };

        console.log(`🔍 EVALUATION FEEDBACK: "${responsePayload.evaluation.feedback}"`);

        if (resultState.finalReport) {
            // ═══════════════════════════════════════════════════════════════
            // INTERVIEW IS OVER
            // ═══════════════════════════════════════════════════════════════
            finalReport = resultState.finalReport;
            responsePayload.done = true;
            responsePayload.finalReport = finalReport;

            const { phraseKeys, uniquePart: lastFbClean } = parseTTSResponse(responsePayload.evaluation.feedback);
            console.log(`🔍 FINAL FEEDBACK - phraseKeys=${JSON.stringify(phraseKeys)}, uniquePart="${lastFbClean}"`);
            console.log(`Converting final feedback to audio...`);

            // Build array of cached audio files to play in sequence
            let cachedAudioSequence = [];

            // Add all cached phrases from feedback (e.g., [thats_okay] [thanks_for_time])
            if (phraseKeys && phraseKeys.length > 0) {
                for (const key of phraseKeys) {
                    const fbFile = checkTTSCache(key);
                    if (fbFile) {
                        console.log(`[TTS] Cache hit for final feedback phrase: ${key}`);
                        cachedAudioSequence.push({
                            key: key,
                            audio: fs.readFileSync(fbFile).toString('base64')
                        });
                    }
                }
            }

            // Generate TTS for unique part if exists
            let feedbackAudio = null;
            if (lastFbClean && lastFbClean.trim().length > 0) {
                const rawAudio = await textToAudio(lastFbClean);
                feedbackAudio = rawAudio?.audio ?? rawAudio;
            }

            // Add outro at the end
            const outroFile = checkTTSCache("interview_outro");
            if (outroFile) {
                console.log(`[TTS] Cache hit for outro: interview_outro`);
                cachedAudioSequence.push({
                    key: "interview_outro",
                    audio: fs.readFileSync(outroFile).toString('base64')
                });
            }

            // Return sequence: [cached1, cached2, ...] → unique TTS → outro
            responsePayload.cachedAudioSequence = cachedAudioSequence;
            responsePayload.feedbackAudio = feedbackAudio;
        } else {
            // ═══════════════════════════════════════════════════════════════
            // INTERVIEW CONTINUES
            // ═══════════════════════════════════════════════════════════════
            responsePayload.done = false;
            responsePayload.nextQuestion = resultState.currentQuestion;
            responsePayload.questionNumber = resultState.questionsAsked;
            responsePayload.difficulty = resultState.difficultyLevel;

            console.log(`🔍 NEXT QUESTION: "${resultState.currentQuestion}"`);

            // ── Parse feedback (should have NO tags for normal evaluation) ──
            const { phraseKey: fbKey, uniquePart: fbUnique } = parseTTSResponse(responsePayload.evaluation.feedback);

            console.log(`🔍 FEEDBACK - phraseKey="${fbKey}", uniquePart="${fbUnique}"`);

            // ── Parse next question (may have [next_question] or [final_question] tag) ──
            const { phraseKey: qKey, uniquePart: questionSpoken } = parseTTSResponse(resultState.currentQuestion);

            console.log(`🔍 QUESTION - phraseKey="${qKey}", uniquePart="${questionSpoken}"`);

            responsePayload.nextQuestion = questionSpoken;

            // ── Build audio ──
            let feedbackCachedAudio = null;
            let questionCachedAudio = null;
            let combinedTextForTTS = "";

            // FEEDBACK: Check for cached phrase (edge cases only)
            if (fbKey) {
                const fbCachedFile = checkTTSCache(fbKey);
                if (fbCachedFile) {
                    console.log(`[TTS] Cache hit for feedback: ${fbKey}`);
                    feedbackCachedAudio = fs.readFileSync(fbCachedFile).toString('base64');
                }
            }

            // FEEDBACK: Add unique part to TTS if exists
            if (fbUnique && fbUnique.trim().length > 0) {
                combinedTextForTTS += fbUnique + " ";
            }

            // QUESTION: Check for cached transition phrase
            if (qKey) {
                const qCachedFile = checkTTSCache(qKey);
                if (qCachedFile) {
                    console.log(`[TTS] Cache hit for question transition: ${qKey}`);
                    questionCachedAudio = fs.readFileSync(qCachedFile).toString('base64');
                }
            }

            // QUESTION: Add unique part to TTS
            if (questionSpoken && questionSpoken.trim().length > 0) {
                combinedTextForTTS += questionSpoken;
            }

            // Generate TTS for combined text
            let combinedAudio = null;
            if (combinedTextForTTS.trim().length > 0) {
                console.log(`[TTS] Generating speech for: "${combinedTextForTTS}"`);
                const ttsResult = await textToAudio(combinedTextForTTS.trim());
                combinedAudio = ttsResult?.audio ?? ttsResult;
            }

            responsePayload.feedbackCachedAudio = feedbackCachedAudio;
            responsePayload.questionCachedAudio = questionCachedAudio;
            responsePayload.combinedAudio = combinedAudio;
        }

        res.json(responsePayload);

    } catch (error) {
        console.error("Interview answer error:", error);
        res.status(500).json({ error: error.message || "Failed to process answer" });
    }
});

// Start the server
httpServer.listen(PORT, () => {
    console.log(`\n========================================`);
    console.log(`🚀 Quiz Server running on http://localhost:${PORT}`);
    console.log(`🔌 Socket.io streaming enabled`);
    console.log(`========================================\n`);
});