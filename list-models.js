import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

async function listModels() {
    console.log("--- Listing Available Gemini Models ---");
    
    if (!process.env.GEMINI_API_KEY) {
        console.error("❌ Error: GEMINI_API_KEY is missing from .env");
        process.exit(1);
    }

    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    
    try {
        // Use the native fetch approach to hit the models endpoint
        const apiKey = process.env.GEMINI_API_KEY;
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);
        const data = await response.json();
        
        if (data.models) {
            console.log("Supported Models:");
            data.models.forEach(model => {
                console.log(`- ${model.name} (${model.displayName})`);
            });
        } else {
            console.log("No models found or error in response:", JSON.stringify(data, null, 2));
        }
    } catch (error) {
        console.error("\n❌ FAILED:", error.message);
    }
}

listModels();
