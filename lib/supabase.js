import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import { serverLog } from './logger.js';
dotenv.config();

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
    serverLog.warn('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY — DB features disabled');
}

// Service role client — for server-side operations (bypasses RLS)
export const supabaseAdmin = createClient(
    supabaseUrl || '',
    supabaseServiceKey || '',
    { auth: { autoRefreshToken: false, persistSession: false } }
);

// Anon client factory — for auth operations scoped to a user's JWT
export function createSupabaseClient(accessToken) {
    return createClient(
        supabaseUrl || '',
        supabaseAnonKey || supabaseServiceKey || '',
        {
            global: {
                headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {}
            }
        }
    );
}
