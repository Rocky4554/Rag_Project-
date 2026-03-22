import dotenv from 'dotenv';
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
        console.error('\n========================================');
        console.error('FATAL: Missing required environment variables:');
        missing.forEach(key => console.error(`   - ${key}`));
        console.error('\nCheck your .env file. See .env.example for reference.');
        console.error('========================================\n');
        process.exit(1);
    }

    // Warn about optional groups
    for (const [group, keys] of Object.entries(OPTIONAL_GROUPS)) {
        const groupMissing = keys.filter(key => !process.env[key]);
        if (groupMissing.length > 0 && groupMissing.length < keys.length) {
            console.warn(`[Env] Partial config for ${group}: missing ${groupMissing.join(', ')}`);
        }
    }
}

export function printEnvStatus() {
    console.log('\n========================================');
    console.log('Environment Status:');
    REQUIRED.forEach(key => {
        console.log(`   ${key.padEnd(25)} : Loaded`);
    });
    for (const [group, keys] of Object.entries(OPTIONAL_GROUPS)) {
        const allPresent = keys.every(key => process.env[key]);
        console.log(`   ${group.padEnd(25)} : ${allPresent ? 'Loaded' : 'Not configured'}`);
    }
    console.log('========================================\n');
}
