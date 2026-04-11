/**
 * livekitAgentSession.js — Modern LiveKit Agents v1.x Voice Pipeline
 *
 * This is the NEW standard way to build voice agents using LiveKit's
 * official Agents SDK (`@livekit/agents` v1.x).
 *
 * Instead of manually wiring:
 *   AudioPublisher → 10ms PCM chunks → LiveKit tracks
 *   Deepgram WebSocket → transcript events → Socket.io
 *   Polly/Deepgram/ElevenLabs TTS → raw PCM → AudioPublisher
 *
 * The new SDK handles ALL of this automatically via `AgentSession`:
 *   - VAD (Voice Activity Detection) via Silero
 *   - STT via Deepgram plugin (or LiveKit Inference)
 *   - LLM via OpenAI/Google/Groq plugins
 *   - TTS via Deepgram/Cartesia/OpenAI plugins
 *   - Barge-in, turn detection, audio streaming — all built-in
 *   - Transcriptions sent to frontend via LiveKit Data Channel (no Socket.io needed)
 *
 * ─── HOW TO USE ───────────────────────────────────────────────────
 *
 * Option A: Standalone Worker (LiveKit dispatches jobs to this process)
 *
 *   import { createStandaloneAgent } from './agents/shared/livekitAgentSession.js';
 *   createStandaloneAgent();
 *
 * Option B: Programmatic (your Express server starts the session manually)
 *
 *   import { LiveKitAgentSession } from './agents/shared/livekitAgentSession.js';
 *   const session = new LiveKitAgentSession({ sessionId, instructions, tools });
 *   await session.start();
 *   // session.stop() when done
 *
 * ─── REQUIRED PACKAGES ───────────────────────────────────────────
 *
 *   npm install @livekit/agents @livekit/agents-plugin-deepgram \
 *              @livekit/agents-plugin-openai @livekit/agents-plugin-silero \
 *              @livekit/agents-plugin-google
 *
 * ─── REQUIRED ENV VARS ───────────────────────────────────────────
 *
 *   LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
 *   DEEPGRAM_API_KEY          (for STT)
 *   OPENAI_API_KEY            (for LLM — or use GEMINI_API_KEY with Google plugin)
 *   DEEPGRAM_API_KEY          (for TTS — or use Cartesia/ElevenLabs/OpenAI)
 *
 * ──────────────────────────────────────────────────────────────────
 */

import dotenv from 'dotenv';
dotenv.config();

// ═══════════════════════════════════════════════════════════════════
// IMPORTS — LiveKit Agents v1.x
// ═══════════════════════════════════════════════════════════════════

// Core framework
import {
    defineAgent,
    cli,
    voice,
    llm,
    ServerOptions,
} from '@livekit/agents';

// Plugins — STT, TTS, VAD, LLM
import * as deepgram from '@livekit/agents-plugin-deepgram';
import * as openai from '@livekit/agents-plugin-openai';
import * as silero from '@livekit/agents-plugin-silero';

// Optional: Google Gemini plugin (for Gemini LLM or Realtime)
// import * as google from '@livekit/agents-plugin-google';

import { AccessToken } from 'livekit-server-sdk';
import { agentLog } from '../../lib/logger.js';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION — Select providers via env vars
// ═══════════════════════════════════════════════════════════════════

const STT_MODEL = process.env.LK_STT_MODEL || 'nova-3';
const STT_LANGUAGE = process.env.LK_STT_LANGUAGE || 'en';
const LLM_MODEL = process.env.LK_LLM_MODEL || 'gpt-4o-mini';
const LLM_PROVIDER = process.env.LK_LLM_PROVIDER || 'openai'; // 'openai' | 'google'
const TTS_MODEL = process.env.LK_TTS_MODEL || 'aura-2-asteria-en';
const TTS_PROVIDER = process.env.LK_TTS_PROVIDER || 'deepgram'; // 'deepgram' | 'openai'

// ═══════════════════════════════════════════════════════════════════
// FACTORY FUNCTIONS — Create STT, LLM, TTS instances
// ═══════════════════════════════════════════════════════════════════

/**
 * Create the STT (Speech-to-Text) instance.
 * Deepgram Nova-3 is the recommended default for accuracy + speed.
 */
function createSTT() {
    return new deepgram.STT({
        model: STT_MODEL,
        language: STT_LANGUAGE,
        smartFormat: true,
        fillerWords: true,
        punctuate: true,
    });
}

/**
 * Create the LLM instance.
 * Supports OpenAI and Google Gemini via plugins.
 */
function createLLM(modelOverride) {
    const model = modelOverride || LLM_MODEL;

    if (LLM_PROVIDER === 'google') {
        // Requires: npm install @livekit/agents-plugin-google
        // Uncomment the import at the top and use:
        // return new google.LLM({ model });
        throw new Error('Google LLM plugin requires @livekit/agents-plugin-google. Uncomment the import and install the package.');
    }

    // Default: OpenAI
    return new openai.LLM({ model });
}

/**
 * Create the TTS (Text-to-Speech) instance.
 * Supports Deepgram Aura and OpenAI TTS.
 */
function createTTS() {
    if (TTS_PROVIDER === 'openai') {
        return new openai.TTS({
            model: process.env.LK_TTS_OPENAI_MODEL || 'tts-1',
            voice: process.env.LK_TTS_OPENAI_VOICE || 'alloy',
        });
    }

    // Default: Deepgram
    return new deepgram.TTS({
        model: TTS_MODEL,
    });
}

// ═══════════════════════════════════════════════════════════════════
// REUSABLE AGENT CLASS — Extend voice.Agent with custom tools
// ═══════════════════════════════════════════════════════════════════

/**
 * BaseVoiceAgent — A reusable voice agent class.
 *
 * Extend this class to create custom voice agents with tools,
 * custom instructions, and event handlers.
 *
 * @example
 * class MyAgent extends BaseVoiceAgent {
 *   constructor() {
 *     super({
 *       instructions: "You are a customer support agent...",
 *       tools: {
 *         lookupOrder: llm.tool({
 *           description: "Look up an order by ID",
 *           parameters: z.object({ orderId: z.string() }),
 *           execute: async ({ orderId }) => { ... },
 *         }),
 *       },
 *     });
 *   }
 * }
 */
export class BaseVoiceAgent extends voice.Agent {
    constructor({
        instructions = 'You are a helpful voice AI assistant. Respond concisely and conversationally.',
        tools = {},
    } = {}) {
        super({ instructions, tools });
    }
}

// ═══════════════════════════════════════════════════════════════════
// PROGRAMMATIC SESSION — Start from your Express server
// ═══════════════════════════════════════════════════════════════════

/**
 * LiveKitAgentSession — Programmatically create and manage a voice agent.
 *
 * Use this when you want to start a voice agent from your existing
 * Express server (e.g., when a user clicks "Start Voice Chat"),
 * instead of running a separate LiveKit Worker process.
 *
 * @example
 * const agent = new LiveKitAgentSession({
 *   sessionId: 'room-abc-123',
 *   instructions: 'You help users understand their uploaded PDFs.',
 *   tools: {
 *     searchPdf: llm.tool({
 *       description: 'Search the uploaded PDF',
 *       parameters: z.object({ query: z.string() }),
 *       execute: async ({ query }) => {
 *         const docs = await vectorStore.similaritySearch(query, 3);
 *         return docs.map(d => d.pageContent).join('\n\n');
 *       },
 *     }),
 *   },
 * });
 *
 * await agent.start();
 * // ... later
 * agent.stop();
 */
export class LiveKitAgentSession {
    /**
     * @param {Object} opts
     * @param {string} opts.sessionId       - LiveKit room name
     * @param {string} [opts.instructions]  - System prompt for the agent
     * @param {Object} [opts.tools]         - LLM tools (function calling)
     * @param {string} [opts.identity]      - Agent participant identity
     * @param {string} [opts.displayName]   - Agent display name in the room
     * @param {string} [opts.greeting]      - Initial greeting to speak
     * @param {Object} [opts.io]            - Socket.io instance (optional, for extra events)
     * @param {string} [opts.llmModel]      - Override LLM model
     */
    constructor({
        sessionId,
        instructions,
        tools = {},
        identity = 'voice-agent',
        displayName = 'Voice Agent',
        greeting = 'Hello! How can I help you today?',
        io = null,
        llmModel = null,
    }) {
        this.sessionId = sessionId;
        this.instructions = instructions || 'You are a helpful voice AI assistant.';
        this.tools = tools;
        this.identity = identity;
        this.displayName = displayName;
        this.greeting = greeting;
        this.io = io;
        this.llmModel = llmModel;

        this.session = null;
        this.room = null;
        this.isActive = false;
    }

    /**
     * Generate a LiveKit access token for the agent to join the room.
     */
    async _generateToken() {
        const at = new AccessToken(
            process.env.LIVEKIT_API_KEY,
            process.env.LIVEKIT_API_SECRET,
            { identity: this.identity, name: this.displayName }
        );
        at.addGrant({
            roomJoin: true,
            room: this.sessionId,
            canPublish: true,
            canSubscribe: true,
        });
        return await at.toJwt();
    }

    /**
     * Start the voice agent session.
     *
     * This:
     * 1. Loads the Silero VAD model
     * 2. Creates the STT → LLM → TTS pipeline
     * 3. Connects to the LiveKit room
     * 4. Starts listening to the user and responding
     * 5. Optionally speaks an initial greeting
     */
    async start() {
        const startTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, 'LiveKitAgentSession starting');

        try {
            // 1. Load VAD model
            agentLog.info({ sessionId: this.sessionId }, '[1/4] Loading Silero VAD...');
            const vad = await silero.VAD.load();

            // 2. Create the Agent with instructions and tools
            agentLog.info({ sessionId: this.sessionId }, '[2/4] Creating Agent...');
            const agent = new BaseVoiceAgent({
                instructions: this.instructions,
                tools: this.tools,
            });

            // 3. Create the AgentSession with all pipeline components
            agentLog.info({ sessionId: this.sessionId, stt: STT_MODEL, llm: this.llmModel || LLM_MODEL, tts: TTS_MODEL }, '[3/4] Creating AgentSession pipeline...');
            this.session = new voice.AgentSession({
                stt: createSTT(),
                llm: createLLM(this.llmModel),
                tts: createTTS(),
                vad: vad,
            });

            // Wire up session events for logging / Socket.io bridge
            this._setupSessionEvents();

            // 4. Connect to LiveKit room
            agentLog.info({ sessionId: this.sessionId }, '[4/4] Connecting to LiveKit room...');
            const { Room } = await import('@livekit/rtc-node');
            this.room = new Room();
            const token = await this._generateToken();
            await this.room.connect(process.env.LIVEKIT_URL, token);

            // Start the session
            await this.session.start({
                agent,
                room: this.room,
            });

            this.isActive = true;
            const totalMs = Math.round(performance.now() - startTs);
            agentLog.info({ sessionId: this.sessionId, totalMs }, 'LiveKitAgentSession ready');

            // 5. Speak initial greeting
            if (this.greeting) {
                this.session.generateReply({
                    instructions: this.greeting,
                });
            }
        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'LiveKitAgentSession start failed');
            throw err;
        }
    }

    /**
     * Wire up AgentSession events for logging and optional Socket.io bridging.
     */
    _setupSessionEvents() {
        if (!this.session) return;

        // User speech transcription
        this.session.on('userTranscription', (transcription) => {
            agentLog.info({ sessionId: this.sessionId, text: transcription.text?.substring(0, 150) }, '🎤 User said');
            if (this.io) {
                this.io.to(this.sessionId).emit('voice_transcript', { role: 'user', text: transcription.text });
            }
        });

        // Agent speech transcription
        this.session.on('agentTranscription', (transcription) => {
            agentLog.info({ sessionId: this.sessionId, text: transcription.text?.substring(0, 150) }, '🤖 Agent said');
            if (this.io) {
                this.io.to(this.sessionId).emit('voice_transcript', { role: 'ai', text: transcription.text });
            }
        });

        // Agent state changes (listening, thinking, speaking)
        this.session.on('agentStateChanged', (state) => {
            agentLog.debug({ sessionId: this.sessionId, state }, 'Agent state changed');
            if (this.io) {
                this.io.to(this.sessionId).emit('voice_state', { state });
            }
        });

        // Tool calls
        this.session.on('toolCallStarted', (toolCall) => {
            agentLog.info({ sessionId: this.sessionId, tool: toolCall.name }, 'Tool call started');
        });

        this.session.on('toolCallCompleted', (toolCall) => {
            agentLog.info({ sessionId: this.sessionId, tool: toolCall.name }, 'Tool call completed');
        });

        // Errors
        this.session.on('error', (err) => {
            agentLog.error({ sessionId: this.sessionId, err: err.message }, 'AgentSession error');
        });

        // Session close
        this.session.on('close', () => {
            agentLog.info({ sessionId: this.sessionId }, 'AgentSession closed');
            this.isActive = false;
        });
    }

    /**
     * Send a message to the agent programmatically (text injection).
     * Useful for triggering agent behavior from server-side events.
     */
    sendMessage(text) {
        if (this.session) {
            this.session.generateReply({ instructions: text });
        }
    }

    /**
     * Stop the voice agent session and disconnect from the room.
     */
    stop() {
        if (!this.isActive) return;
        agentLog.info({ sessionId: this.sessionId }, 'LiveKitAgentSession stopping');
        this.isActive = false;

        try { this.session?.close(); } catch (_) {}
        try { this.room?.disconnect(); } catch (_) {}
    }
}

// ═══════════════════════════════════════════════════════════════════
// STANDALONE WORKER — Run as a separate LiveKit Agent process
// ═══════════════════════════════════════════════════════════════════

/**
 * createStandaloneAgent — Create a standalone LiveKit Agent Worker.
 *
 * This is the standard LiveKit approach where the agent runs as its own
 * process and LiveKit's server dispatches "jobs" to it when users join rooms.
 *
 * To run:
 *   node agents/shared/livekitAgentSession.js dev
 *
 * @param {Object} [opts]
 * @param {string} [opts.instructions]  - Default system prompt
 * @param {Object} [opts.tools]         - Default tools for all sessions
 * @param {string} [opts.agentName]     - Worker name for LiveKit
 */
export function createStandaloneAgent({
    instructions = 'You are a helpful, friendly voice AI assistant. Respond concisely and conversationally.',
    tools = {},
    agentName = 'rag-voice-agent',
} = {}) {
    const agentDef = defineAgent({
        // Prewarm: Load heavy models once when the worker starts (shared across jobs)
        prewarm: async (proc) => {
            agentLog.info('Prewarming: Loading Silero VAD model...');
            proc.userData.vad = await silero.VAD.load();
            agentLog.info('Prewarm complete');
        },

        // Entry: Called for each new room/job
        entry: async (ctx) => {
            agentLog.info({ room: ctx.room?.name }, 'Agent job received');

            // Create the Agent with instructions and tools
            const agent = new BaseVoiceAgent({ instructions, tools });

            // Create the pipeline session
            const session = new voice.AgentSession({
                stt: createSTT(),
                llm: createLLM(),
                tts: createTTS(),
                vad: ctx.proc.userData.vad,
            });

            // Start the session
            await session.start({
                agent,
                room: ctx.room,
            });

            // Connect to the room
            await ctx.connect();

            agentLog.info({ room: ctx.room?.name }, 'Agent connected and listening');

            // Greet the user
            session.generateReply({
                instructions: 'Greet the user warmly and let them know you are ready to help.',
            });
        },
    });

    // Start the CLI runner
    cli.runApp(
        new ServerOptions({
            agent: import.meta.url,
            agentName,
        })
    );

    return agentDef;
}

// ═══════════════════════════════════════════════════════════════════
// GENERATE CLIENT TOKEN — For your frontend to join the same room
// ═══════════════════════════════════════════════════════════════════

/**
 * Generate a LiveKit access token for a frontend client.
 *
 * @param {string} roomName   - The room to join (matches the agent's sessionId)
 * @param {string} identity   - Unique user identity
 * @param {string} [name]     - Display name
 * @returns {Promise<string>} JWT token
 *
 * @example
 * // In your Express route:
 * app.post('/api/voice/token', async (req, res) => {
 *   const token = await generateClientToken(req.body.roomName, req.user.id, req.user.name);
 *   res.json({ token, livekitUrl: process.env.LIVEKIT_URL });
 * });
 */
export async function generateClientToken(roomName, identity, name = 'User') {
    const at = new AccessToken(
        process.env.LIVEKIT_API_KEY,
        process.env.LIVEKIT_API_SECRET,
        { identity, name }
    );
    at.addGrant({
        roomJoin: true,
        room: roomName,
        canPublish: true,
        canSubscribe: true,
    });
    return await at.toJwt();
}

// ═══════════════════════════════════════════════════════════════════
// AUTO-RUN — If this file is executed directly, start standalone
// ═══════════════════════════════════════════════════════════════════

// If run directly: node agents/shared/livekitAgentSession.js dev
const isMainModule = process.argv[1] &&
    (process.argv[1].endsWith('livekitAgentSession.js') ||
     process.argv[1].endsWith('livekitAgentSession'));

if (isMainModule) {
    createStandaloneAgent();
}
