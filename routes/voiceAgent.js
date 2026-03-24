import { Router } from 'express';
import { AccessToken } from 'livekit-server-sdk';
import { VoiceAgentWorker } from '../agents/voice/worker.js';
import { optionalAuth } from '../middleware/auth.js';
import { ensureSession } from '../lib/sessionRestore.js';
import { agentLog } from '../lib/logger.js';

export function createVoiceAgentRoutes({ sessionCache, activeVoiceAgents, io }) {
    const router = Router();

    // POST /api/voice-agent/start
    router.post('/voice-agent/start', optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

            const livekitUrl = process.env.LIVEKIT_URL;
            if (!livekitUrl) return res.status(500).json({ error: 'LiveKit not configured' });

            // Try to restore session (for PDF context), but allow voice agent without a document
            await ensureSession(sessionCache, sessionId).catch(() => {});

            agentLog.info({ sessionId, userId: req.user?.id, type: 'voice' }, 'Voice agent start requested');

            // Stop existing agent if running
            if (activeVoiceAgents.has(sessionId)) {
                activeVoiceAgents.get(sessionId).stop();
                activeVoiceAgents.delete(sessionId);
            }

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
                agentLog.error({ sessionId, err: err.message, type: 'voice' }, 'Voice agent start error');
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

            agentLog.info({ sessionId, type: 'voice' }, 'Voice agent started');
            res.json({ token, url: livekitUrl, roomName: sessionId });

        } catch (error) {
            agentLog.error({ err: error.message, type: 'voice' }, 'Voice agent start error');
            res.status(500).json({ error: error.message || 'Failed to start voice agent' });
        }
    });

    // POST /api/voice-agent/stop
    router.post('/voice-agent/stop', optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

            if (activeVoiceAgents.has(sessionId)) {
                activeVoiceAgents.get(sessionId).stop();
                activeVoiceAgents.delete(sessionId);
                agentLog.info({ sessionId, type: 'voice' }, 'Voice agent stopped');
            }

            res.json({ stopped: true });
        } catch (error) {
            agentLog.error({ err: error.message, type: 'voice' }, 'Voice agent stop error');
            res.status(500).json({ error: error.message || 'Failed to stop voice agent' });
        }
    });

    return router;
}
