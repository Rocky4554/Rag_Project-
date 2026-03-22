import { Router } from 'express';
import bcrypt from 'bcryptjs';
import { supabaseAdmin } from '../lib/supabase.js';
import { signAccessToken, signRefreshToken, verifyToken, cookieOptions } from '../lib/jwt.js';
import { sendOTPEmail, generateOTP } from '../lib/email.js';
import { requireAuth } from '../middleware/auth.js';
import { getUserDocuments, getRecentActivity } from '../lib/db.js';
import { validate, authSignupSchema, authLoginSchema, authForgotSchema, authVerifyOtpSchema, authResetPasswordSchema } from '../lib/validation.js';
import { authLog } from '../lib/logger.js';

const ACCESS_MAX_AGE = 7 * 24 * 60 * 60 * 1000;   // 7 days
const REFRESH_MAX_AGE = 30 * 24 * 60 * 60 * 1000;  // 30 days

function setAuthCookies(res, userId, email, name) {
    const payload = { id: userId, email, name: name || '' };
    const accessToken = signAccessToken(payload);
    const refreshToken = signRefreshToken({ id: userId });

    res.cookie('access_token', accessToken, cookieOptions(ACCESS_MAX_AGE));
    res.cookie('refresh_token', refreshToken, cookieOptions(REFRESH_MAX_AGE));

    return { accessToken, refreshToken };
}

export function createAuthRoutes() {
    const router = Router();

    // ── Sign Up ─────────────────────────────────────────────────────
    router.post('/auth/signup', validate(authSignupSchema), async (req, res) => {
        try {
            const { email, password, name } = req.validated;

            // Check if email already exists
            const { data: existing } = await supabaseAdmin
                .from('users')
                .select('id')
                .eq('email', email.toLowerCase())
                .single();

            if (existing) return res.status(400).json({ error: 'Email already registered' });

            // Hash password
            const password_hash = await bcrypt.hash(password, 12);

            // Create user
            const { data: user, error } = await supabaseAdmin
                .from('users')
                .insert({ email: email.toLowerCase(), password_hash, name: name || '' })
                .select('id, email, name')
                .single();

            if (error) throw error;

            setAuthCookies(res, user.id, user.email, user.name);

            authLog.info({ email }, 'User signed up');
            res.json({ user: { id: user.id, email: user.email, name: user.name } });
        } catch (error) {
            authLog.error({ err: error.message }, 'Signup error');
            res.status(500).json({ error: 'Signup failed' });
        }
    });

    // ── Login ───────────────────────────────────────────────────────
    router.post('/auth/login', validate(authLoginSchema), async (req, res) => {
        try {
            const { email, password } = req.validated;

            const { data: user, error } = await supabaseAdmin
                .from('users')
                .select('id, email, name, password_hash')
                .eq('email', email.toLowerCase())
                .single();

            if (error || !user) return res.status(401).json({ error: 'Invalid email or password' });

            const valid = await bcrypt.compare(password, user.password_hash);
            if (!valid) return res.status(401).json({ error: 'Invalid email or password' });

            setAuthCookies(res, user.id, user.email, user.name);

            authLog.info({ email }, 'User logged in');
            res.json({ user: { id: user.id, email: user.email, name: user.name } });
        } catch (error) {
            authLog.error({ err: error.message }, 'Login error');
            res.status(500).json({ error: 'Login failed' });
        }
    });

    // ── Refresh Token ───────────────────────────────────────────────
    router.post('/auth/refresh', async (req, res) => {
        try {
            const token = req.cookies?.refresh_token;
            if (!token) return res.status(401).json({ error: 'No refresh token' });

            const decoded = verifyToken(token);

            const { data: user } = await supabaseAdmin
                .from('users')
                .select('id, email, name')
                .eq('id', decoded.id)
                .single();

            if (!user) return res.status(401).json({ error: 'User not found' });

            setAuthCookies(res, user.id, user.email, user.name);
            res.json({ user: { id: user.id, email: user.email, name: user.name } });
        } catch (error) {
            authLog.error({ err: error.message }, 'Refresh error');
            res.status(401).json({ error: 'Invalid refresh token' });
        }
    });

    // ── Get Profile (authenticated) ─────────────────────────────────
    router.get('/auth/profile', requireAuth, async (req, res) => {
        try {
            const [documents, recentActivity] = await Promise.all([
                getUserDocuments(req.user.id),
                getRecentActivity(req.user.id, 5)
            ]);

            res.json({
                user: { id: req.user.id, email: req.user.email, name: req.user.name },
                documents,
                recentActivity
            });
        } catch (error) {
            authLog.error({ err: error.message }, 'Profile error');
            res.status(500).json({ error: 'Failed to load profile' });
        }
    });

    // ── Logout ──────────────────────────────────────────────────────
    router.post('/auth/logout', (req, res) => {
        res.clearCookie('access_token', { path: '/' });
        res.clearCookie('refresh_token', { path: '/' });
        res.json({ message: 'Logged out' });
    });

    // ── Forgot Password — send OTP ──────────────────────────────────
    router.post('/auth/forgot-password', validate(authForgotSchema), async (req, res) => {
        try {
            const { email } = req.validated;

            const { data: user } = await supabaseAdmin
                .from('users')
                .select('id, name')
                .eq('email', email.toLowerCase())
                .single();

            // Always return success to prevent email enumeration
            if (!user) return res.json({ message: 'If that email exists, a reset code has been sent.' });

            // Invalidate previous OTPs
            await supabaseAdmin
                .from('password_resets')
                .update({ used: true })
                .eq('user_id', user.id)
                .eq('used', false);

            // Generate & store OTP
            const otp = generateOTP();
            const expiresAt = new Date(Date.now() + 10 * 60 * 1000); // 10 min

            await supabaseAdmin.from('password_resets').insert({
                user_id: user.id,
                otp,
                expires_at: expiresAt.toISOString(),
            });

            // Send email
            await sendOTPEmail(email.toLowerCase(), otp, user.name);

            authLog.info({ email }, 'OTP sent');
            res.json({ message: 'If that email exists, a reset code has been sent.' });
        } catch (error) {
            authLog.error({ err: error.message }, 'Forgot password error');
            res.status(500).json({ error: 'Failed to send reset code' });
        }
    });

    // ── Verify OTP ──────────────────────────────────────────────────
    router.post('/auth/verify-otp', validate(authVerifyOtpSchema), async (req, res) => {
        try {
            const { email, otp } = req.validated;

            const { data: user } = await supabaseAdmin
                .from('users')
                .select('id')
                .eq('email', email.toLowerCase())
                .single();

            if (!user) return res.status(400).json({ error: 'Invalid email or code' });

            const { data: reset } = await supabaseAdmin
                .from('password_resets')
                .select('*')
                .eq('user_id', user.id)
                .eq('otp', otp)
                .eq('used', false)
                .gte('expires_at', new Date().toISOString())
                .order('created_at', { ascending: false })
                .limit(1)
                .single();

            if (!reset) return res.status(400).json({ error: 'Invalid or expired code' });

            res.json({ valid: true });
        } catch (error) {
            authLog.error({ err: error.message }, 'Verify OTP error');
            res.status(500).json({ error: 'Verification failed' });
        }
    });

    // ── Reset Password ──────────────────────────────────────────────
    router.post('/auth/reset-password', validate(authResetPasswordSchema), async (req, res) => {
        try {
            const { email, otp, newPassword } = req.validated;

            const { data: user } = await supabaseAdmin
                .from('users')
                .select('id')
                .eq('email', email.toLowerCase())
                .single();

            if (!user) return res.status(400).json({ error: 'Invalid request' });

            // Verify OTP one more time
            const { data: reset } = await supabaseAdmin
                .from('password_resets')
                .select('id')
                .eq('user_id', user.id)
                .eq('otp', otp)
                .eq('used', false)
                .gte('expires_at', new Date().toISOString())
                .order('created_at', { ascending: false })
                .limit(1)
                .single();

            if (!reset) return res.status(400).json({ error: 'Invalid or expired code' });

            // Update password
            const password_hash = await bcrypt.hash(newPassword, 12);
            await supabaseAdmin
                .from('users')
                .update({ password_hash })
                .eq('id', user.id);

            // Mark OTP as used
            await supabaseAdmin
                .from('password_resets')
                .update({ used: true })
                .eq('id', reset.id);

            authLog.info({ email }, 'Password reset successful');
            res.json({ message: 'Password reset successfully' });
        } catch (error) {
            authLog.error({ err: error.message }, 'Reset password error');
            res.status(500).json({ error: 'Password reset failed' });
        }
    });

    return router;
}
