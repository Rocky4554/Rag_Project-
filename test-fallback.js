import { createLLMWithFallback } from "./lib/llm.js";
import { HumanMessage } from "@langchain/core/messages";
import dotenv from "dotenv";

dotenv.config();

async function testFallback() {
    console.log("--- Testing LLM Fallback Mechanism ---");

    // Scenario: Primary provider (Groq) has an invalid key
    // We intentionally override the environment variable for this test process
    process.env.GROQ_API_KEY = "invalid_key_for_testing";

    console.log("Checking environment variables...");
    console.log("OPENROUTER_API_KEY exists:", !!process.env.OPENROUTER_API_KEY);
    if (process.env.OPENROUTER_API_KEY) {
        console.log("OPENROUTER_API_KEY (first 5 chars):", process.env.OPENROUTER_API_KEY.substring(0, 5));
    }

    if (!process.env.OPENROUTER_API_KEY) {
        console.error("❌ Error: OPENROUTER_API_KEY is missing from .env");
        // Instead of exiting, let's list keys to see what's there
        console.log("Available keys:", Object.keys(process.env).filter(k => k.includes("API")));
        process.exit(1);
    }

    const llm = createLLMWithFallback({
        provider: "groq",
        temperature: 0.7,
        maxTokens: 50
    });

    console.log("Sending request (expecting fallback to OpenRouter)...");
    
    try {
        const response = await llm.invoke([
            new HumanMessage("Say 'Fallback Successful' and nothing else.")
        ]);
        
        console.log("\n✅ Response received!");
        console.log("Content:", response.content);
        
        if (response.content.toLowerCase().includes("fallback successful")) {
            console.log("\n✨ VERIFICATION PASSED: The system successfully fell back to OpenRouter/DeepSeek.");
        } else {
            console.log("\n⚠️ VERIFICATION INCOMPLETE: Response received but content unexpected. Check if DeepSeek is responding correctly.");
        }
    } catch (error) {
        console.error("\n❌ VERIFICATION FAILED:", error.message);
        if (error.message.includes("401") || error.message.includes("API key")) {
            console.log("Suggestion: Make sure your OPENROUTER_API_KEY is valid in .env");
        }
    }
}

testFallback();
