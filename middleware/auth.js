import { verifyToken } from '../lib/jwt.js';

/**
 * Auth middleware — extracts user from JWT cookie.
 * Attaches req.user = { id, email, name } on success.
 */
export async function requireAuth(req, res, next) {
    const token = req.cookies?.access_token;
    if (!token) {
        return res.status(401).json({ error: 'Not authenticated' });
    }

    try {
        const decoded = verifyToken(token);
        req.user = { id: decoded.id, email: decoded.email, name: decoded.name };
        next();
    } catch (err) {
        return res.status(401).json({ error: 'Invalid or expired token' });
    }
}

/**
 * Optional auth — attaches req.user if cookie present, but doesn't block.
 */
export async function optionalAuth(req, res, next) {
    const token = req.cookies?.access_token;
    if (!token) {
        req.user = null;
        return next();
    }

    try {
        const decoded = verifyToken(token);
        req.user = { id: decoded.id, email: decoded.email, name: decoded.name };
    } catch {
        req.user = null;
    }
    next();
}
