import { supabaseAdmin } from '../lib/supabase.js';

/**
 * Auth middleware — extracts user from Supabase JWT.
 * Attaches req.user = { id, email, ... } on success.
 * Returns 401 if no valid token.
 */
export async function requireAuth(req, res, next) {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Missing or invalid authorization header' });
    }

    const token = authHeader.split(' ')[1];

    try {
        const { data: { user }, error } = await supabaseAdmin.auth.getUser(token);
        if (error || !user) {
            return res.status(401).json({ error: 'Invalid or expired token' });
        }
        req.user = user;
        next();
    } catch (err) {
        console.error('[Auth Middleware]', err.message);
        return res.status(401).json({ error: 'Authentication failed' });
    }
}

/**
 * Optional auth — attaches req.user if token present, but doesn't block.
 * Useful for endpoints that work with or without auth.
 */
export async function optionalAuth(req, res, next) {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        req.user = null;
        return next();
    }

    const token = authHeader.split(' ')[1];
    try {
        const { data: { user }, error } = await supabaseAdmin.auth.getUser(token);
        req.user = error ? null : user;
    } catch {
        req.user = null;
    }
    next();
}
