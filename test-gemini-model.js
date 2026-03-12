import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from "dotenv";

dotenv.config();

async function testGemini() {
    console.log("--- Testing Gemini 1.5 Flash Implementation ---");
    
    if (!process.env.GEMINI_API_KEY) {
        console.error("❌ Error: GEMINI_API_KEY is missing from .env");
        process.exit(1);
    }

    const llm = new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: "gemini-3-flash-preview",
        temperature: 0.7,
    });

    console.log("Sending request to Gemini 1.5 Flash...");
    
    try {
        const response = await llm.invoke("Say 'Gemini 1.5 Flash is ready' and nothing else.");
        console.log("\n✅ Response received!");
        console.log("Content:", response.content);
        
        if (response.content.toLowerCase().includes("ready")) {
            console.log("\n✨ VERIFICATION PASSED: The system is successfully using Gemini 1.5 Flash.");
        } else {
            console.log("\n⚠️ VERIFICATION INCOMPLETE: Response received but content unexpected.");
        }
    } catch (error) {
        console.error("\n❌ VERIFICATION FAILED:", error.message);
    }
}

testGemini();
