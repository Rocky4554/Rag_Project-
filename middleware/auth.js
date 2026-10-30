import { verifyAccessToken } from '../lib/jwt.js';
import { isTokenCurrent } from '../lib/passwordVersion.js';

/**
 * Decode the access-token cookie. Returns the user, or null when the token is
 * invalid, expired, or was issued before a password reset.
 */
async function userFromToken(token) {
    let decoded;
    try {
        decoded = verifyAccessToken(token);
    } catch {
        return null;
    }
    if (!(await isTokenCurrent(decoded))) return null;
    return { id: decoded.id, email: decoded.email, name: decoded.name };
}

/**
 * Auth middleware — extracts user from JWT cookie.
 * Attaches req.user = { id, email, name } on success.
 */
export async function requireAuth(req, res, next) {
    const token = req.cookies?.access_token;
    if (!token) {
        return res.status(401).json({ error: 'Not authenticated' });
    }

    const user = await userFromToken(token);
    if (!user) return res.status(401).json({ error: 'Invalid or expired token' });
    req.user = user;
    next();
}

/**
 * Optional auth — attaches req.user if a cookie is present; anonymous requests pass through.
 * A cookie that is present but expired/invalid gets a 401 so the client refreshes and
 * retries, instead of silently acting as anonymous (owned documents would 404, chats go unsaved).
 */
export async function optionalAuth(req, res, next) {
    const token = req.cookies?.access_token;
    if (!token) {
        req.user = null;
        return next();
    }

    const user = await userFromToken(token);
    if (!user) return res.status(401).json({ error: 'Invalid or expired token' });
    req.user = user;
    next();
}
