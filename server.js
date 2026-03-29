import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import cookieParser from 'cookie-parser';
import rateLimit from 'express-rate-limit';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';

import { validateEnv, printEnvStatus } from './lib/env.js';
import { serverLog, httpLogger } from './lib/logger.js';
import { registerSocket, unregisterSocket, createInterviewAgent } from "./lib/interview/interviewAgent.js";
import { createUploadRoutes } from './routes/upload.js';
import { createChatRoutes } from './routes/chat.js';
import { createQuizRoutes } from './routes/quiz.js';
import { createSummaryRoutes } from './routes/summary.js';
import { createTokenRoutes } from './routes/tokens.js';
import { createInterviewRoutes } from './routes/interview.js';
import { createAuthRoutes } from './routes/auth.js';
import { createHistoryRoutes } from './routes/history.js';
import { createSessionRoutes } from './routes/session.js';
import { createVoiceAgentRoutes } from './routes/voiceAgent.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Validate environment before anything else ────────────────────
validateEnv();
printEnvStatus();

// ── App setup ─────────────────────────────────────────────────────
const app = express();
const httpServer = createServer(app);
const PORT = process.env.PORT || 3000;

const allowedOrigins = process.env.CORS_ORIGINS
    ? process.env.CORS_ORIGINS.split(',').map(s => s.trim())
    : [`http://localhost:${PORT}`];

const io = new SocketIOServer(httpServer, {
    cors: { origin: allowedOrigins, methods: ["GET", "POST"], credentials: true }
});

// ── Shared state ─────────────────────────────────────────────────
const sessionCache = {};
const activeAgents = new Map();
const activeVoiceAgents = new Map();
const clientReadyResolvers = new Map();

// ── Session TTL cleanup (prevents memory leak) ──────────────────
const SESSION_TTL = parseInt(process.env.SESSION_TTL_MS) || 2 * 60 * 60 * 1000; // 2 hours default

setInterval(() => {
    const now = Date.now();
    let cleaned = 0;
    for (const [id, session] of Object.entries(sessionCache)) {
        if (now - (session.createdAt || 0) > SESSION_TTL) {
            // Stop active agents if running
            if (activeAgents.has(id)) {
                activeAgents.get(id).stop();
                activeAgents.delete(id);
            }
            if (activeVoiceAgents.has(id)) {
                activeVoiceAgents.get(id).stop();
                activeVoiceAgents.delete(id);
            }
            delete sessionCache[id];
            cleaned++;
        }
    }
    if (cleaned > 0) {
        serverLog.info({ cleaned, remaining: Object.keys(sessionCache).length }, 'Session cleanup');
    }
}, 10 * 60 * 1000); // check every 10 minutes

// ── Socket.io ────────────────────────────────────────────────────
io.on('connection', (socket) => {
    serverLog.debug({ socketId: socket.id }, 'Client connected');

    socket.on('register_session', (sessionId) => {
        if (sessionId && typeof sessionId === 'string') {
            socket.join(sessionId);
            socket.data.sessionId = sessionId;
            registerSocket(sessionId, socket);
            serverLog.debug({ socketId: socket.id, sessionId }, 'Session registered');
        }
    });

    socket.on('client_audio_ready', (sessionId) => {
        serverLog.debug({ sessionId }, 'Client audio ready');
        const resolve = clientReadyResolvers.get(sessionId);
        if (resolve) {
            clientReadyResolvers.delete(sessionId);
            resolve();
        }
    });

    socket.on('disconnect', () => {
        if (socket.data.sessionId) {
            unregisterSocket(socket.data.sessionId);
        }
        serverLog.debug({ socketId: socket.id }, 'Client disconnected');
    });
});

// ── Security middleware ──────────────────────────────────────────
// Trust Railway's proxy so rate-limiter uses real client IP (X-Forwarded-For)
app.set('trust proxy', 1);
app.use(helmet({
    contentSecurityPolicy: false, // allow inline scripts in public/index.html
    crossOriginEmbedderPolicy: false
}));
app.use(cors({ origin: allowedOrigins, methods: ['GET', 'POST', 'DELETE'], credentials: true }));
app.use(cookieParser());
app.use(express.json({ limit: '5mb' }));
app.use(httpLogger);
app.use(express.static(path.join(__dirname, 'public')));

// ── Health check (before all route handlers) ────────────────────
app.get('/ping', (req, res) => {
    res.json({ status: 'ok', message: 'pong', timestamp: new Date().toISOString() });
});

// ── Rate limiters ────────────────────────────────────────────────
const apiLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 60,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests, please try again later' }
});

const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100,
    message: { error: 'Too many auth attempts, please try again later' }
});

const llmLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 20,
    message: { error: 'Too many AI requests, please slow down' }
});

const uploadLimiter = rateLimit({
    windowMs: 5 * 60 * 1000, // 5 minutes
    max: 10,
    message: { error: 'Too many uploads, please try again later' }
});

// ── Multer (memory storage — no files written to disk) ──────────
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 20 * 1024 * 1024 }, // 20MB max
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/pdf') cb(null, true);
        else cb(new Error('Only PDF files are allowed!'), false);
    }
});

// ── Global error handler ─────────────────────────────────────────
function errorHandler(err, req, res, _next) {
    serverLog.error({ err: err.message, path: req.path }, 'Unhandled error');
    const status = err.status || 500;
    res.status(status).json({
        error: process.env.NODE_ENV === 'production' ? 'Internal server error' : err.message
    });
}

// ── Initialize & Start ──────────────────────────────────────────
async function startServer() {
    // Initialize the interview agent with PostgresSaver (async)
    serverLog.info('Initializing interview agent...');
    let interviewAgent;
    try {
        interviewAgent = await createInterviewAgent();
        serverLog.info('Interview agent ready (PostgresSaver)');
    } catch (err) {
        serverLog.warn({ err: err.message }, 'PostgresSaver failed, using no checkpointer');
        interviewAgent = await createInterviewAgent(null);
    }

    // ── Routes with rate limiters ────────────────────────────────
    const deps = { sessionCache, activeAgents, activeVoiceAgents, clientReadyResolvers, io, upload, interviewAgent };

    app.use('/api/auth', authLimiter);
    app.use('/api', createAuthRoutes());

    app.use('/api/upload', uploadLimiter);
    app.use('/api', createUploadRoutes(deps));

    app.use('/api/chat', llmLimiter);
    app.use('/api', createChatRoutes(deps));

    app.use('/api/quiz', llmLimiter);
    app.use('/api', createQuizRoutes(deps));

    app.use('/api/summary', llmLimiter);
    app.use('/api', createSummaryRoutes(deps));

    app.use('/api/interview', llmLimiter);
    app.use('/api', createInterviewRoutes(deps));

    app.use('/api/voice-agent', llmLimiter);
    app.use('/api', createVoiceAgentRoutes(deps));

    app.use('/api', apiLimiter);
    app.use('/api', createTokenRoutes());
    app.use('/api', createHistoryRoutes());
    app.use('/api', createSessionRoutes(deps));

    // Global error handler (must be last)
    app.use(errorHandler);

    // ── Start ────────────────────────────────────────────────────
    httpServer.listen(PORT, () => {
        serverLog.info({
            port: PORT,
            supabase: !!process.env.SUPABASE_URL,
            langsmith: process.env.LANGCHAIN_TRACING_V2 === 'true',
            sessionTTL: `${SESSION_TTL / 60000}min`
        }, 'Server started');
    });
}

startServer().catch(err => {
    serverLog.fatal({ err: err.message }, 'Fatal startup error');
    process.exit(1);
});
