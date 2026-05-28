import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { agentLog } from "../../lib/logger.js";

dotenv.config();

/**
 * Synchronous, zero-cost rule-based transcript cleanup.
 *
 * Rules applied in order:
 *   1. Strip leading/trailing whitespace
 *   2. Collapse filler words (uh, um, er, erm, hmm, hm)
 *   3. Collapse consecutive word repeats (STT stuttering: "I I want" → "I want")
 *   4. Capitalize first letter
 *   5. Add terminal period if missing (text > 10 chars)
 *
 * @param {string} text - Raw transcript text
 * @returns {string} Cleaned transcript text
 */
export function cleanTranscript(text) {
    if (typeof text !== "string") return "";

    // 1. Strip leading/trailing whitespace
    let cleaned = text.trim();

    // 2. Collapse filler words surrounded by word boundaries, then collapse extra spaces
    cleaned = cleaned.replace(/\b(uh|um|er|erm|hmm|hm)\b/gi, "");
    cleaned = cleaned.replace(/\s{2,}/g, " ").trim();

    // 3. Collapse consecutive word repeats (applied twice for triple+ repeats)
    cleaned = cleaned.replace(/\b(\w+)\s+\1\b/gi, "$1");
    cleaned = cleaned.replace(/\b(\w+)\s+\1\b/gi, "$1");

    // If cleaning produced empty string, return original trimmed input unchanged
    if (cleaned === "") return text.trim();

    // 4. Capitalize first letter
    if (cleaned.length > 0 && cleaned[0] !== cleaned[0].toUpperCase()) {
        cleaned = cleaned[0].toUpperCase() + cleaned.slice(1);
    }

    // 5. Add terminal period if missing (length > 10, not ending in .?!,  and not ending with ...)
    if (
        cleaned.length > 10 &&
        !cleaned.endsWith("...") &&
        !/[.?!,]$/.test(cleaned)
    ) {
        cleaned = cleaned + ".";
    }

    return cleaned;
}

/**
 * Async Gemini Flash check for semantic completeness.
 *
 * Called ONLY for short transcripts (< 8 words, no terminal punctuation)
 * before routing to the LLM to verify it forms a complete thought.
 *
 * On any error or timeout (> 2 seconds), returns true (assume complete).
 *
 * @param {string} text - Short/ambiguous transcript text
 * @returns {Promise<boolean>} true if complete thought, false if not
 */
export async function checkSemanticCompleteness(text) {
    const prompt = `Does this sentence form a complete thought? Reply only YES or NO.\n\nSentence: "${text}"`;

    try {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            agentLog.warn({ text }, "checkSemanticCompleteness: GEMINI_API_KEY not set, assuming complete");
            return true;
        }

        const genAI = new GoogleGenerativeAI(apiKey);
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

        // Race the Gemini call against a 2-second timeout
        const timeoutPromise = new Promise((_resolve, reject) =>
            setTimeout(() => reject(new Error("checkSemanticCompleteness: 2s timeout")), 2000)
        );

        const geminiPromise = model.generateContent(prompt);

        const result = await Promise.race([geminiPromise, timeoutPromise]);
        const responseText = result.response.text().trim();

        // If response contains "NO" (case-insensitive) and not "YES", incomplete
        const upper = responseText.toUpperCase();
        const isComplete = !(upper.includes("NO") && !upper.includes("YES"));

        agentLog.info(
            { text, responseText, isComplete },
            "checkSemanticCompleteness: Gemini response received"
        );

        return isComplete;
    } catch (err) {
        agentLog.warn(
            { text, err: err.message },
            "checkSemanticCompleteness: failed or timed out, assuming complete"
        );
        return true;
    }
}
