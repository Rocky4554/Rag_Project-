import { Router } from 'express';
import { AccessToken, RoomServiceClient } from 'livekit-server-sdk';
import { VoiceAgentWorker } from '../agents/voice/worker.js';
import { optionalAuth } from '../middleware/auth.js';
import { ensureSession } from '../lib/sessionRestore.js';
import { agentLog } from '../lib/logger.js';

// LiveKit Room Service client for server-side room/participant management
const roomService = (process.env.LIVEKIT_URL && process.env.LIVEKIT_API_KEY && process.env.LIVEKIT_API_SECRET)
    ? new RoomServiceClient(
        process.env.LIVEKIT_URL.replace('wss://', 'https://'),
        process.env.LIVEKIT_API_KEY,
        process.env.LIVEKIT_API_SECRET
    )
    : null;

/**
 * Forcibly clean up all agents and their LiveKit participants for a session.
 * Best practice per LiveKit docs: use RoomServiceClient.removeParticipant()
 * to immediately disconnect stale agent participants from the room, then
 * stop the in-process worker.
 */
async function cleanupSessionAgents(sessionId, { activeAgents, activeVoiceAgents }) {
    const cleanups = [];

    // Stop interview agent worker + kick its LiveKit participant
    if (activeAgents.has(sessionId)) {
        const agent = activeAgents.get(sessionId);
        agentLog.info({ sessionId }, 'Cleaning up active interview agent');
        agent.stop();
        activeAgents.delete(sessionId);
        if (roomService) {
            cleanups.push(
                roomService.removeParticipant(sessionId, 'ai-interviewer')
                    .catch(() => {}) // may already be gone
            );
        }
    }

    // Stop voice agent worker + kick its LiveKit participant
    if (activeVoiceAgents.has(sessionId)) {
        const agent = activeVoiceAgents.get(sessionId);
        agentLog.info({ sessionId }, 'Cleaning up active voice agent');
        agent.stop();
        activeVoiceAgents.delete(sessionId);
        if (roomService) {
            cleanups.push(
                roomService.removeParticipant(sessionId, 'voice-agent')
                    .catch(() => {})
            );
        }
    }

    // Wait for LiveKit to process the removals
    if (cleanups.length > 0) {
        await Promise.all(cleanups);
        // Small delay to let LiveKit propagate disconnection before new agent joins
        await new Promise(r => setTimeout(r, 500));
    }
}

export { cleanupSessionAgents };

export function createConversationalAiRoutes({ sessionCache, activeAgents, activeVoiceAgents, io }) {
    const router = Router();

    // POST /api/conversational-ai/start
    router.post('/conversational-ai/start', optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

            const livekitUrl = process.env.LIVEKIT_URL;
            if (!livekitUrl) return res.status(500).json({ error: 'LiveKit not configured' });

            // Try to restore session (for PDF context), but allow voice agent without a document
            await ensureSession(sessionCache, sessionId).catch(() => {});

            agentLog.info({ sessionId, userId: req.user?.id, type: 'voice' }, 'Conversational AI start requested');

            // Clean up ALL existing agents on this session before starting a new one
            await cleanupSessionAgents(sessionId, { activeAgents, activeVoiceAgents });

            // Create and start the voice agent worker (connects LiveKit + Gemini Live)
            const worker = new VoiceAgentWorker(
                sessionId,
                sessionCache,
                io,
                req.user?.id || null,
                req.user?.name || null
            );
            activeVoiceAgents.set(sessionId, worker);

            // Start agent in background — don't block HTTP response
            worker.start().catch(err => {
                agentLog.error({ sessionId, err: err.message, type: 'voice' }, 'Conversational AI start error');
                activeVoiceAgents.delete(sessionId);
                io.to(sessionId).emit('voice_error', { error: err.message });
            });

            // Generate LiveKit token for the browser client
            const at = new AccessToken(process.env.LIVEKIT_API_KEY, process.env.LIVEKIT_API_SECRET, {
                identity: `user-${sessionId}`,
                name: req.user?.name || 'User',
                ttl: '2h'
            });
            at.addGrant({
                roomJoin: true,
                room: sessionId,
                canPublish: true,
                canSubscribe: true
            });
            const token = await at.toJwt();

            agentLog.info({ sessionId, type: 'voice' }, 'Conversational AI started');
            res.json({ token, url: livekitUrl, roomName: sessionId });

        } catch (error) {
            agentLog.error({ err: error.message, type: 'voice' }, 'Voice agent start error');
            res.status(500).json({ error: error.message || 'Failed to start Conversational AI' });
        }
    });

    // POST /api/conversational-ai/stop
    router.post('/conversational-ai/stop', optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

            if (activeVoiceAgents.has(sessionId)) {
                activeVoiceAgents.get(sessionId).stop();
                activeVoiceAgents.delete(sessionId);
                agentLog.info({ sessionId, type: 'voice' }, 'Conversational AI stopped');
            }

            // Also kick the voice-agent participant from LiveKit room
            if (roomService) {
                roomService.removeParticipant(sessionId, 'voice-agent').catch(() => {});
            }

            res.json({ stopped: true });
        } catch (error) {
            agentLog.error({ err: error.message, type: 'voice' }, 'Voice agent stop error');
            res.status(500).json({ error: error.message || 'Failed to stop voice agent' });
        }
    });

    return router;
}
