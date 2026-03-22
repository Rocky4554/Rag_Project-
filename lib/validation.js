import { z } from 'zod';

// ── Route schemas ────────────────────────────────────────────────

export const uploadSchema = z.object({
    // file validated by multer, nothing in body required
});

export const chatSchema = z.object({
    sessionId: z.string().min(1, 'Session ID is required'),
    question: z.string().min(1, 'Question is required').max(2000, 'Question too long')
});

export const quizSchema = z.object({
    sessionId: z.string().min(1, 'Session ID is required'),
    topic: z.string().max(200).optional(),
    numQuestions: z.union([z.string(), z.number()]).optional()
});

export const summarySchema = z.object({
    sessionId: z.string().min(1, 'Session ID is required')
});

export const interviewStartSchema = z.object({
    sessionId: z.string().min(1, 'Session ID is required'),
    maxQuestions: z.union([z.string(), z.number()]).optional()
});

export const authSignupSchema = z.object({
    email: z.string().email('Invalid email'),
    password: z.string().min(6, 'Password must be at least 6 characters'),
    name: z.string().max(100).optional()
});

export const authLoginSchema = z.object({
    email: z.string().email('Invalid email'),
    password: z.string().min(1, 'Password is required')
});

export const authRefreshSchema = z.object({
    refresh_token: z.string().min(1, 'Refresh token is required')
});

export const authForgotSchema = z.object({
    email: z.string().email('Invalid email')
});

export const authVerifyOtpSchema = z.object({
    email: z.string().email('Invalid email'),
    otp: z.string().length(6, 'OTP must be 6 digits')
});

export const authResetPasswordSchema = z.object({
    email: z.string().email('Invalid email'),
    otp: z.string().length(6, 'OTP must be 6 digits'),
    newPassword: z.string().min(6, 'Password must be at least 6 characters')
});

// ── Validation middleware factory ─────────────────────────────────

export function validate(schema) {
    return (req, res, next) => {
        const result = schema.safeParse(req.body);
        if (!result.success) {
            const errors = result.error.errors.map(e => e.message);
            return res.status(400).json({ error: errors[0], details: errors });
        }
        req.validated = result.data;
        next();
    };
}
