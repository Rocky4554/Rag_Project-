import { Router } from 'express';
import { AccessToken } from 'livekit-server-sdk';
import { serverLog } from '../lib/logger.js';

export function createTokenRoutes() {
    const router = Router();

    // LiveKit token
    router.post('/livekit/token', async (req, res) => {
        try {
            const { sessionId } = req.body;
            if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });

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
            res.status(500).json({ error: error.message || 'Failed to generate LiveKit token' });
        }
    });

    // Deepgram token
    router.get('/deepgram/token', async (req, res) => {
        try {
            const apiKey = process.env.DEEPGRAM_API_KEY;
            if (!apiKey) {
                return res.status(500).json({ error: 'DEEPGRAM_API_KEY not configured' });
            }

            const model = process.env.DEEPGRAM_STT_MODEL || 'nova-3';
            const language = process.env.DEEPGRAM_STT_LANGUAGE || 'en';
            const smartFormat = (process.env.DEEPGRAM_STT_SMART_FORMAT || 'true').toLowerCase() === 'true';
            const interimResults = (process.env.DEEPGRAM_STT_INTERIM_RESULTS || 'true').toLowerCase() === 'true';
            const endpointing = parseInt(process.env.DEEPGRAM_STT_ENDPOINTING_MS || '500', 10);

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
            serverLog.error({ err: error.message }, 'Deepgram token generation error');
            res.status(500).json({ error: 'Failed to retrieve Deepgram token' });
        }
    });

    return router;
}
