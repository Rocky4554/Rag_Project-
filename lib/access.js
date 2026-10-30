/**
 * Session access control. A session uploaded while logged in belongs to that
 * user (ownerId); anonymous uploads (ownerId null) stay reachable by sessionId alone.
 */
export function canAccessSession(session, user) {
    if (!session?.ownerId) return true;
    return user?.id === session.ownerId;
}

/**
 * Error text that is safe to send to clients — hides internal details in production.
 */
export function publicError(err, fallback) {
    if (process.env.NODE_ENV === 'production') return fallback;
    return err?.message || fallback;
}
