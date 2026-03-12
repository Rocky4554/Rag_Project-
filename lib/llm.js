import { ChatGroq } from "@langchain/groq";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";

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

    if (provider === "groq") {
        primary = new ChatGroq({
            apiKey: process.env.GROQ_API_KEY,
            model: "llama-3.3-70b-versatile",
            temperature,
            maxTokens,
            streaming,
            callbacks,
        });
    } else if (provider === "google") {
        primary = new ChatGoogleGenerativeAI({
            apiKey: process.env.GEMINI_API_KEY,
            model: "gemini-3-flash-preview",
            temperature,
            maxOutputTokens: maxTokens,
            streaming,
            callbacks,
        });
    } else {
        throw new Error(`Unsupported primary provider: ${provider}`);
    }

    // fallback to OpenRouter DeepSeek R1
    const backup = new ChatOpenAI({
        openAIApiKey: process.env.OPENROUTER_API_KEY,
        modelName: "deepseek/deepseek-r1:free",
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

    console.log(`[LLM Factory] Initialized ${provider} with OpenRouter fallback.`);

    return primary.withFallbacks({
        fallbacks: [backup],
    });
}
