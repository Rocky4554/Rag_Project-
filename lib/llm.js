import { ChatGroq } from "@langchain/groq";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
import { llmLog } from "./logger.js";

dotenv.config();

/**
 * Creates an LLM instance with a fallback to OpenRouter (DeepSeek R1).
 *
 * @param {Object} options - LLM configuration options.
 * @param {string} options.provider - Primary provider ('groq' or 'google').
 * @param {number} [options.temperature=0.7] - model temperature.
 * @param {number} [options.maxTokens] - max tokens to generate.
 * @param {boolean} [options.streaming=false] - whether to stream responses.
 * @param {Array} [options.callbacks=[]] - optional LangChain callbacks.
 */
export function createLLMWithFallback({
    provider,
    temperature = 0.7,
    maxTokens,
    streaming = false,
    callbacks = []
}) {
    let primary;
    let primaryModel;

    if (provider === "groq") {
        primaryModel = "llama-3.3-70b-versatile";
        primary = new ChatGroq({
            apiKey: process.env.GROQ_API_KEY,
            model: primaryModel,
            temperature,
            maxTokens,
            streaming,
            callbacks,
        });
    } else if (provider === "google") {
        primaryModel = "gemini-2.5-flash-lite";
        primary = new ChatGoogleGenerativeAI({
            apiKey: process.env.GEMINI_API_KEY,
            model: primaryModel,
            temperature,
            maxOutputTokens: maxTokens,
            streaming,
            callbacks,
        });
    } else {
        throw new Error(`Unsupported primary provider: ${provider}`);
    }

    const backupModel = "meta-llama/llama-3.3-70b-instruct";
    // fallback to OpenRouter Llama model
    const backup = new ChatOpenAI({
        openAIApiKey: process.env.OPENROUTER_API_KEY,
        modelName: backupModel,
        temperature,
        maxTokens,
        streaming,
        callbacks,
    }, {
        baseURL: "https://openrouter.ai/api/v1",
        defaultHeaders: {
            "HTTP-Referer": "https://github.com/Rocky4554/Rag_Project", // Optional, for OpenRouter rankings
            "X-Title": "RAG Interview Project",
        },
    });

    llmLog.info({ provider, primaryModel, backupModel, temperature, maxTokens, streaming }, 'LLM factory initialized');

    return primary.withFallbacks({
        fallbacks: [backup],
    });
}

/**
 * Detects rate-limit or token-exhaustion errors from LLM responses/errors.
 * Returns a descriptive string if detected, null otherwise.
 */
export function detectRateLimitError(err) {
    if (!err) return null;
    const msg = (err.message || '').toLowerCase();
    const status = err.status || err.statusCode || err.response?.status;

    if (status === 429) return 'rate_limit_exceeded';
    if (status === 503) return 'service_overloaded';
    if (msg.includes('rate limit') || msg.includes('rate_limit')) return 'rate_limit_exceeded';
    if (msg.includes('quota') || msg.includes('exhausted')) return 'token_quota_exhausted';
    if (msg.includes('too many requests')) return 'rate_limit_exceeded';
    if (msg.includes('resource_exhausted') || msg.includes('resource exhausted')) return 'token_quota_exhausted';
    if (msg.includes('tokens per min') || msg.includes('tokens per day')) return 'token_quota_exhausted';
    if (msg.includes('capacity') || msg.includes('overloaded')) return 'service_overloaded';
    return null;
}

