import dotenv from 'dotenv';
import { serverLog } from './logger.js';
dotenv.config();

const REQUIRED = [
    'GEMINI_API_KEY',
    'GROQ_API_KEY',
    'QDRANT_URL',
    'QDRANT_API_KEY',
];

const OPTIONAL_GROUPS = {
    'Supabase (auth + persistence)': ['SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_DB_URL'],
    'LiveKit (voice interviews)': ['LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_URL'],
    'AWS Polly (TTS)': ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'],
    'Deepgram (STT)': ['DEEPGRAM_API_KEY'],
    'LangSmith (tracing)': ['LANGCHAIN_API_KEY'],
};

export function validateEnv() {
    const missing = REQUIRED.filter(key => !process.env[key]);
    if (missing.length > 0) {
        serverLog.fatal({ missing }, 'Missing required environment variables');
        process.exit(1);
    }

    // Warn about optional groups
    for (const [group, keys] of Object.entries(OPTIONAL_GROUPS)) {
        const groupMissing = keys.filter(key => !process.env[key]);
        if (groupMissing.length > 0 && groupMissing.length < keys.length) {
            serverLog.warn({ group, missing: groupMissing }, 'Partial env config');
        }
    }
}

export function printEnvStatus() {
    const status = {};
    REQUIRED.forEach(key => {
        status[key] = 'Loaded';
    });
    for (const [group, keys] of Object.entries(OPTIONAL_GROUPS)) {
        const allPresent = keys.every(key => process.env[key]);
        status[group] = allPresent ? 'Loaded' : 'Not configured';
    }
    serverLog.info({ envStatus: status }, 'Environment status');
}
