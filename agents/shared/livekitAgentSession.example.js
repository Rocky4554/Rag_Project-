/**
 * livekitAgentSession.example.js — Usage Examples
 *
 * Shows how to use the modern LiveKit AgentSession in different scenarios.
 * This file is NOT meant to be run directly — it's a reference guide.
 */

import { llm } from '@livekit/agents';
import { z } from 'zod';
import { LiveKitAgentSession, generateClientToken } from './livekitAgentSession.js';

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 1: Simple Voice Chat (no tools, no RAG)
// ═══════════════════════════════════════════════════════════════════

async function example_simpleVoiceChat() {
    const agent = new LiveKitAgentSession({
        sessionId: 'room-simple-123',
        instructions: 'You are a friendly assistant. Keep responses short and conversational.',
        greeting: 'Hey there! What can I help you with?',
    });

    await agent.start();
    // Agent is now listening in LiveKit room "room-simple-123"
    // Frontend connects using generateClientToken('room-simple-123', 'user-id')
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 2: RAG Voice Agent with PDF Search (like your current app)
// ═══════════════════════════════════════════════════════════════════

async function example_ragVoiceAgent(sessionId, sessionCache, io) {
    const session = sessionCache[sessionId];

    const agent = new LiveKitAgentSession({
        sessionId,
        io, // Optional: bridge transcripts to Socket.io too
        instructions: `You are a helpful voice assistant. The user has uploaded a document called "${session?.originalName || 'a PDF'}".
Use the search_pdf tool to find information from the document when needed.
Keep responses concise — 1 to 3 sentences for voice.`,
        greeting: `Hello! I've loaded your document. Ask me anything about it!`,
        tools: {
            search_pdf: llm.tool({
                description: "Search the user's uploaded PDF document for specific information",
                parameters: z.object({
                    query: z.string().describe('Search query to find relevant content in the PDF'),
                }),
                execute: async ({ query }) => {
                    if (!session?.vectorStore) {
                        return 'No document is currently loaded.';
                    }
                    const docs = await session.vectorStore.similaritySearch(query, 3);
                    return docs.map(d => d.pageContent).join('\n\n') || 'No relevant content found.';
                },
            }),
        },
    });

    await agent.start();
    return agent; // Return so the caller can stop it later
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 3: Interview Agent with LangGraph Integration
// ═══════════════════════════════════════════════════════════════════

async function example_interviewAgent(sessionId, sessionCache, agentWorkflow, candidateName) {
    const agent = new LiveKitAgentSession({
        sessionId,
        instructions: `You are an AI interviewer conducting a voice interview with ${candidateName}.
Ask questions based on the document they uploaded.
Evaluate their answers and provide brief feedback before the next question.
Keep a professional but friendly tone.`,
        tools: {
            processTurn: llm.tool({
                description: 'Process the candidate answer through the interview state machine',
                parameters: z.object({
                    answer: z.string().describe("The candidate's answer to evaluate"),
                }),
                execute: async ({ answer }) => {
                    // Bridge to your existing LangGraph workflow
                    const result = await agentWorkflow.invoke({
                        answer,
                        sessionId,
                    });
                    return JSON.stringify({
                        feedback: result.evaluation?.feedback,
                        nextQuestion: result.nextQuestion,
                        score: result.evaluation?.score,
                        done: result.done,
                    });
                },
            }),
        },
    });

    await agent.start();
    return agent;
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 4: Express Route Integration
// ═══════════════════════════════════════════════════════════════════

/**
 * How to add voice endpoints to your Express server:
 *
 * In routes/voice.js:
 */
function example_expressRoute(app, sessionCache, io) {
    // Endpoint: Start a voice session
    app.post('/api/voice/start', async (req, res) => {
        const { sessionId, userId, userName } = req.body;

        try {
            const agent = new LiveKitAgentSession({
                sessionId,
                io,
                instructions: `You are a helpful voice assistant for ${userName}.`,
                greeting: `Hello ${userName}! How can I help you today?`,
            });

            await agent.start();

            // Store reference for cleanup
            sessionCache[sessionId] = sessionCache[sessionId] || {};
            sessionCache[sessionId]._voiceAgent = agent;

            // Generate token for the frontend
            const clientToken = await generateClientToken(sessionId, userId, userName);

            res.json({
                token: clientToken,
                livekitUrl: process.env.LIVEKIT_URL,
            });
        } catch (err) {
            res.status(500).json({ error: err.message });
        }
    });

    // Endpoint: Stop a voice session
    app.post('/api/voice/stop', (req, res) => {
        const { sessionId } = req.body;
        const agent = sessionCache[sessionId]?._voiceAgent;
        if (agent) {
            agent.stop();
            delete sessionCache[sessionId]._voiceAgent;
        }
        res.json({ ok: true });
    });
}

// ═══════════════════════════════════════════════════════════════════
// EXAMPLE 5: Frontend (React) — Connecting to the voice agent
// ═══════════════════════════════════════════════════════════════════

/**
 * In your React app:
 *
 * ```jsx
 * import { LiveKitRoom, RoomAudioRenderer, useVoiceAssistant } from '@livekit/components-react';
 *
 * function VoiceChat({ token, livekitUrl }) {
 *   return (
 *     <LiveKitRoom serverUrl={livekitUrl} token={token} connect={true}>
 *       <RoomAudioRenderer />
 *       <VoiceUI />
 *     </LiveKitRoom>
 *   );
 * }
 *
 * function VoiceUI() {
 *   const { state, audioTrack, agentTranscriptions } = useVoiceAssistant();
 *
 *   return (
 *     <div>
 *       <p>Status: {state}</p> {// 'listening' | 'thinking' | 'speaking' }
 *
 *       {// Transcriptions are streamed automatically via LiveKit data channel!
 *       // No Socket.io needed for transcripts! }
 *       {agentTranscriptions.map((t, i) => (
 *         <p key={i}>{t.text}</p>
 *       ))}
 *     </div>
 *   );
 * }
 * ```
 *
 * Key difference from the old approach:
 *   OLD: Socket.io emits 'voice_transcript' events manually
 *   NEW: LiveKit sends transcriptions via its built-in data channel automatically
 */
