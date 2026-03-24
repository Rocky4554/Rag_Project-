/**
 * Lists Gemini models available to your API key that support bidiGenerateContent (Live API).
 * Run: node scripts/listLiveModels.js
 */
import dotenv from 'dotenv';
dotenv.config();

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
    console.error('GEMINI_API_KEY not set in .env');
    process.exit(1);
}

async function listModels(apiVersion) {
    const url = `https://generativelanguage.googleapis.com/${apiVersion}/models?key=${API_KEY}`;
    const res = await fetch(url);
    if (!res.ok) {
        console.error(`[${apiVersion}] HTTP ${res.status}: ${await res.text()}`);
        return [];
    }
    const data = await res.json();
    return data.models || [];
}

async function main() {
    console.log('Checking which models support bidiGenerateContent (Live API)...\n');

    for (const version of ['v1beta', 'v1alpha']) {
        console.log(`--- API version: ${version} ---`);
        try {
            const models = await listModels(version);
            const liveModels = models.filter(m =>
                m.supportedGenerationMethods?.includes('bidiGenerateContent')
            );

            if (liveModels.length === 0) {
                console.log('  No models found with bidiGenerateContent support.\n');
            } else {
                for (const m of liveModels) {
                    // Strip "models/" prefix for use in code
                    const shortName = m.name.replace(/^models\//, '');
                    console.log(`  ✓ ${shortName}  (display: ${m.displayName || '?'})`);
                }
                console.log();
            }
        } catch (err) {
            console.error(`  Error: ${err.message}\n`);
        }
    }

    console.log('Usage: set GEMINI_LIVE_MODEL=<model_name> in your .env');
    console.log('       set GEMINI_LIVE_API_VERSION=v1beta or v1alpha accordingly\n');
}

main();
