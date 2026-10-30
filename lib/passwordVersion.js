import { createHash } from 'crypto';
import { supabaseAdmin } from './supabase.js';
import { authLog } from './logger.js';

/**
 * Password fingerprint embedded in auth tokens (`pwv`). A password reset changes the
 * hash, so tokens issued before it stop matching — that's how a reset logs out other
 * sessions without a token-version column.
 */
export function passwordVersion(passwordHash) {
    return createHash('sha256').update(passwordHash || '').digest('hex').substring(0, 16);
}

const CACHE_TTL_MS = 60 * 1000;
const cache = new Map(); // userId → { pwv, at }

/** Drop the cached fingerprint (call after changing a user's password). */
export function forgetPasswordVersion(userId) {
    cache.delete(userId);
}

/**
 * Whether a decoded token was issued for the user's current password.
 * Tokens from before `pwv` existed are rejected — otherwise they would survive a
 * reset (and a refresh would mint fresh tokens from them); costs one re-login.
 * Fails open on DB errors so a database blip doesn't log everyone out.
 */
export async function isTokenCurrent(decoded) {
    if (!decoded?.pwv) return false;
    const hit = cache.get(decoded.id);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.pwv === decoded.pwv;
    try {
        const { data: user, error } = await supabaseAdmin
            .from('users')
            .select('password_hash')
            .eq('id', decoded.id)
            .single();
        // PGRST116 = no row: the user is gone. Any other error is the DB, not the token.
        if (error?.code === 'PGRST116') return false;
        if (error || !user) throw new Error(error?.message || 'user lookup returned nothing');
        const pwv = passwordVersion(user.password_hash);
        cache.set(decoded.id, { pwv, at: Date.now() });
        return pwv === decoded.pwv;
    } catch (err) {
        authLog.warn({ err: err.message }, 'Password version check failed — allowing token');
        return true;
    }
}
