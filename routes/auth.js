import { Router } from 'express';
import { createSupabaseClient, supabaseAdmin } from '../lib/supabase.js';
import { requireAuth } from '../middleware/auth.js';
import { getUserDocuments, getRecentActivity } from '../lib/db.js';
import { validate, authSignupSchema, authLoginSchema, authRefreshSchema } from '../lib/validation.js';
import { authLog } from '../lib/logger.js';

export function createAuthRoutes() {
    const router = Router();

    // ── Sign Up ─────────────────────────────────────────────────────
    router.post('/auth/signup', validate(authSignupSchema), async (req, res) => {
        try {
            const { email, password, name } = req.validated;

            // Use admin API to create user (auto-confirms, bypasses email rate limits)
            const { data: userData, error: createError } = await supabaseAdmin.auth.admin.createUser({
                email,
                password,
                email_confirm: true,
                user_metadata: { name: name || '' }
            });

            if (createError) return res.status(400).json({ error: createError.message });

            // Sign in immediately to get a session
            const supabase = createSupabaseClient();
            const { data: signInData, error: signInError } = await supabase.auth.signInWithPassword({
                email,
                password
            });

            if (signInError) return res.status(400).json({ error: signInError.message });

            authLog.info({ email }, 'User signed up');
            res.json({
                user: { id: userData.user.id, email: userData.user.email, name: userData.user.user_metadata?.name },
                session: signInData.session
            });
        } catch (error) {
            authLog.error({ err: error.message }, 'Signup error');
            res.status(500).json({ error: 'Signup failed' });
        }
    });

    // ── Login ───────────────────────────────────────────────────────
    router.post('/auth/login', validate(authLoginSchema), async (req, res) => {
        try {
            const { email, password } = req.validated;

            const supabase = createSupabaseClient();
            const { data, error } = await supabase.auth.signInWithPassword({ email, password });

            if (error) return res.status(401).json({ error: error.message });

            authLog.info({ email }, 'User logged in');
            res.json({
                user: { id: data.user.id, email: data.user.email, name: data.user.user_metadata?.name },
                session: data.session
            });
        } catch (error) {
            authLog.error({ err: error.message }, 'Login error');
            res.status(500).json({ error: 'Login failed' });
        }
    });

    // ── Refresh Token ───────────────────────────────────────────────
    router.post('/auth/refresh', validate(authRefreshSchema), async (req, res) => {
        try {
            const { refresh_token } = req.validated;

            const supabase = createSupabaseClient();
            const { data, error } = await supabase.auth.refreshSession({ refresh_token });

            if (error) return res.status(401).json({ error: error.message });

            res.json({ session: data.session });
        } catch (error) {
            authLog.error({ err: error.message }, 'Refresh error');
            res.status(500).json({ error: 'Token refresh failed' });
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
                user: {
                    id: req.user.id,
                    email: req.user.email,
                    name: req.user.user_metadata?.name
                },
                documents,
                recentActivity
            });
        } catch (error) {
            authLog.error({ err: error.message }, 'Profile error');
            res.status(500).json({ error: 'Failed to load profile' });
        }
    });

    // ── Logout ──────────────────────────────────────────────────────
    router.post('/auth/logout', async (req, res) => {
        res.json({ message: 'Logged out' });
    });

    return router;
}
