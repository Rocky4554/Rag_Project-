import jwt from 'jsonwebtoken';

const ACCESS_EXPIRY = '7d';
const REFRESH_EXPIRY = '30d';

// Read lazily so dotenv has run; validateEnv() enforces it at startup.
function secret() {
    const s = process.env.JWT_SECRET;
    if (!s) throw new Error('JWT_SECRET is not set');
    return s;
}

export function signAccessToken(payload) {
    return jwt.sign({ ...payload, type: 'access' }, secret(), { expiresIn: ACCESS_EXPIRY });
}

export function signRefreshToken(payload) {
    return jwt.sign({ ...payload, type: 'refresh' }, secret(), { expiresIn: REFRESH_EXPIRY });
}

// Tokens issued before the `type` claim existed: access tokens carry email, refresh tokens don't.
function tokenType(decoded) {
    return decoded.type || (decoded.email ? 'access' : 'refresh');
}

export function verifyAccessToken(token) {
    const decoded = jwt.verify(token, secret());
    if (tokenType(decoded) !== 'access') throw new Error('Not an access token');
    return decoded;
}

export function verifyRefreshToken(token) {
    const decoded = jwt.verify(token, secret());
    if (tokenType(decoded) !== 'refresh') throw new Error('Not a refresh token');
    return decoded;
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
