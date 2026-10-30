import { Router } from 'express';
import { AccessToken } from 'livekit-server-sdk';
import { optionalAuth } from '../middleware/auth.js';
import { ensureSession } from '../lib/sessionRestore.js';
import { canAccessSession, publicError } from '../lib/access.js';
import { serverLog } from '../lib/logger.js';

export function createTokenRoutes({ sessionCache }) {
    const router = Router();

    // LiveKit token
    router.post('/livekit/token', optionalAuth, async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !canAccessSession(session, req.user)) {
                return res.status(404).json({ error: 'Session not found' });
            }

            const apiKey    = process.env.LIVEKIT_API_KEY;
            const apiSecret = process.env.LIVEKIT_API_SECRET;
            const livekitUrl = process.env.LIVEKIT_URL;

            if (!apiKey || !apiSecret || !livekitUrl) {
                return res.status(500).json({ error: 'LiveKit environment variables not configured.' });
            }

            const at = new AccessToken(apiKey, apiSecret, {
                identity: `candidate-${sessionId}`,
                ttl: '2h',
            });

            at.addGrant({
                roomJoin:     true,
                room:         sessionId,
                canPublish:   true,
                canSubscribe: true,
            });

            const token = await at.toJwt();
            serverLog.info({ sessionId }, 'LiveKit token generated');

            res.json({ token, url: livekitUrl });

        } catch (error) {
            serverLog.error({ err: error.message }, 'LiveKit token generation error');
            res.status(500).json({ error: publicError(error, 'Failed to generate LiveKit token') });
        }
    });

    return router;
}
