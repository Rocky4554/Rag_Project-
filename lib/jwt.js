import jwt from 'jsonwebtoken';

const SECRET = process.env.JWT_SECRET || 'rag-learn-secret-change-in-production';
const ACCESS_EXPIRY = '7d';
const REFRESH_EXPIRY = '30d';

export function signAccessToken(payload) {
    return jwt.sign(payload, SECRET, { expiresIn: ACCESS_EXPIRY });
}

export function signRefreshToken(payload) {
    return jwt.sign(payload, SECRET, { expiresIn: REFRESH_EXPIRY });
}

export function verifyToken(token) {
    return jwt.verify(token, SECRET);
}

/** Cookie options for httpOnly secure cookies */
export function cookieOptions(maxAgeMs) {
    const isProd = process.env.NODE_ENV === 'production';
    return {
        httpOnly: true,
        secure: isProd,
        sameSite: isProd ? 'none' : 'lax',
        maxAge: maxAgeMs,
        path: '/',
    };
}
