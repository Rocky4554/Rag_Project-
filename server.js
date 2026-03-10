import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';

import { extractTextFromPDF } from "./lib/pdfLoader.js";
import { splitText } from "./lib/textSplitter.js";
import { storeDocuments } from "./lib/vectorStore.js";
import { generateQuiz } from "./lib/quizGenerator.js";
import { summarizeDocs } from "./lib/summarizer.js";
import { textToAudio } from "./lib/speechToAudio.js";
import { queryRAG } from "./lib/rag.js";
import { createInterviewAgent, registerVectorStore, registerSocket, unregisterSocket, parseTTSResponse, checkTTSCache } from "./lib/interviewAgent.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

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

        // Register the vectorStore in the module-level registry (NOT in graph state - it can't be serialized)
        registerVectorStore(sessionId, session.vectorStore);

        // Initial state — no vectorStore here
        const initialState = {
            sessionId: sessionId,
            maxQuestions: parseInt(maxQuestions),
            difficultyLevel: "medium",
            chatHistory: [],
            questionsAsked: 0,
            topicsUsed: []
        };

        // Run the graph — it pauses at END after generateQuestionNode
        const config = { configurable: { thread_id: sessionId } };
        const resultState = await interviewAgent.invoke(initialState, config);

        // Save the graph state back to our session cache
        session.interviewStateConfig = config; // Keep track of the thread ID

        console.log(`Converting question to audio...`);
        const { phraseKey, uniquePart } = parseTTSResponse(resultState.currentQuestion);

        let introCachedAudio = null;
        const introFile = checkTTSCache("interview_intro");
        if (introFile) {
            console.log(`[TTS] Cache hit for intro: interview_intro`);
            introCachedAudio = fs.readFileSync(introFile).toString('base64');
        }
        let questionAudio;
        const cachedFile = phraseKey ? checkTTSCache(phraseKey) : null;
        if (cachedFile) {
            // Serve the pre-generated cache file as base64
            const buf = fs.readFileSync(cachedFile);
            questionAudio = { audio: buf.toString('base64'), format: 'mp3' };
            console.log(`[TTS] Cache hit: ${phraseKey}`);
        } else {
            questionAudio = await textToAudio(uniquePart);
        }

        res.json({
            question: uniquePart,
            audio: questionAudio?.audio ?? questionAudio,
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

        // Provide the user's answer into the state
        const inputState = {
            userAnswer: answer
        };

        // Resume the graph from where it paused.
        // It will run evaluateAnswer -> adaptNextQuestion -> (generateQuestion OR generateFinalReport) -> END
        const resultState = await interviewAgent.invoke(inputState, session.interviewStateConfig);

        let finalReport = null;
        let responsePayload = {
            evaluation: resultState.evaluation || {
                score: 5, accuracy: 5, clarity: 5, depth: 5,
                feedback: "Thank you for your answer.",
                nextDifficulty: "same"
            }
        };

        if (resultState.finalReport) {
            // Interview is over! Return the report and the evaluation of the last answer.
            finalReport = resultState.finalReport;
            responsePayload.done = true;
            responsePayload.finalReport = finalReport;

            const { phraseKey: lastFbKey, uniquePart: lastFbClean } = parseTTSResponse(responsePayload.evaluation.feedback);
            console.log(`Converting final feedback to audio...`);

            let feedbackAudio = null;
            if (lastFbClean.trim().length > 0) {
                const rawAudio = await textToAudio(lastFbClean);
                feedbackAudio = rawAudio?.audio ?? rawAudio;
            }

            let feedbackCachedAudio = null;
            if (lastFbKey) {
                const fbFile = checkTTSCache(lastFbKey);
                if (fbFile) {
                    console.log(`[TTS] Cache hit for final feedback phrase: ${lastFbKey}`);
                    feedbackCachedAudio = fs.readFileSync(fbFile).toString('base64');
                }
            }

            let outroCachedAudio = null;
            const outroFile = checkTTSCache("interview_outro");
            if (outroFile) {
                console.log(`[TTS] Cache hit for outro: interview_outro`);
                outroCachedAudio = fs.readFileSync(outroFile).toString('base64');
            }

            responsePayload.feedbackCachedAudio = feedbackCachedAudio;
            responsePayload.feedbackAudio = feedbackAudio;
            responsePayload.outroCachedAudio = outroCachedAudio; // Frontend plays feedbackCached -> feedback -> outro
        } else {
            // Interview continues. Return evaluation + next question
            responsePayload.done = false;
            responsePayload.nextQuestion = resultState.currentQuestion;
            responsePayload.questionNumber = resultState.questionsAsked;
            responsePayload.difficulty = resultState.difficultyLevel;

            // ── Build TTS for feedback (use cache for prefix phrase if available) ──
            const { phraseKey: fbKey, uniquePart: fbUnique } = parseTTSResponse(responsePayload.evaluation.feedback);
            const fbCachedFile = fbKey ? checkTTSCache(fbKey) : null;

            // ── Strip tag from question text before TTS ──
            const { uniquePart: questionSpoken } = parseTTSResponse(resultState.currentQuestion);
            responsePayload.nextQuestion = questionSpoken;

            const spokenText = fbUnique + " " + questionSpoken;
            console.log(`Converting feedback and next question to audio...`);
            const ttsAudio = await textToAudio(spokenText);

            if (fbCachedFile) {
                console.log(`[TTS] Cache hit for feedback phrase: ${fbKey}`);
                const buf = fs.readFileSync(fbCachedFile);
                responsePayload.feedbackCachedAudio = buf.toString('base64');
            }
            responsePayload.combinedAudio = ttsAudio?.audio ?? ttsAudio;
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
