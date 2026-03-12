import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';
dotenv.config();

export async function generateQuiz(vectorStore, options = {}) {
    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    const {
        topic = "general concepts",
        numQuestions = 5
    } = options;

    console.log(`Setting up LangChain RAG pipeline for Quiz Generation on topic: "${topic}"`);

    // 1. Initialize the Chat Model (Gemini)
    const llm = new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: "gemini-3-flash-preview",
        temperature: 0.2, // Slightly higher to allow some creativity in distractors
        maxOutputTokens: 8192, // Raised from 2048 — 5 detailed questions can easily exceed 2k tokens
    });

    // 2. Define the System Prompt using LangChain Templates
    // LangChain will automatically inject the retrieved text into the {context} variable
    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are an expert educational content creator. Your task is to extract information from the provided context and create a highly educational multiple-choice quiz.

Context:
{context}

Requirements:
1. Generate exactly {numQuestions} multiple-choice questions based ONLY on the provided context.
2. The questions should focus on the topic: "{topic}". If the topic is broad, cover the most important concepts.
3. Each question must have exactly 4 options (A, B, C, D).
4. Only one option should be correct.
5. Provide a brief explanation for why the correct answer is correct, referencing the context.
6. The output MUST be a valid JSON array of objects.

JSON Format Output Example:
[
  {{
    "question": "What is the main advantage of X?",
    "options": ["A) It is faster", "B) It is cheaper", "C) It is more secure", "D) It is open source"],
    "correctAnswer": 0,
    "explanation": "According to the text, the primary benefit of X is its speed because..."
  }}
]

CRITICAL: Return ONLY the raw JSON array. Do NOT use markdown code fences, backticks, or any wrapper. Start your response with '[' and end with ']'. Any extra text will break the parser.`
        ],
        ["human", "Generate the quiz now."]
    ]);

    // 3. Create the chains
    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: prompt,
    });

    // We retrieve more documents to get a broader context for the quiz
    const retriever = vectorStore.asRetriever({
        k: 10,
    });

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: combineDocsChain,
        retriever: retriever,
    });

    // 4. Invoke the chain!
    // We pass a dummy query to the retriever to fetch relevant docs based on the topic
    console.log(`Generating a ${numQuestions}-question quiz using LangChain and Gemini...`);

    // For retrieval, if the topic is "general", just search for core concepts, else use the topic
    const searchQuery = topic && topic.toLowerCase() !== "general" && topic.trim() !== ""
        ? `information about ${topic} suitable for a quiz`
        : `key concepts, definitions, main ideas, and important facts`;

    const response = await retrievalChain.invoke({
        input: searchQuery,
        topic: topic || "general concepts",
        numQuestions: numQuestions.toString()
    });

    // 5. Parse the LLM output (which should be JSON)
    console.log("Parsing Gemini's JSON response...");
    try {
        let rawAnswer = response.answer;
        // Sometimes LLMs still try to wrap the JSON despite instructions
        if (rawAnswer.startsWith("```json")) {
            rawAnswer = rawAnswer.substring(7);
        }
        if (rawAnswer.startsWith("```")) {
            rawAnswer = rawAnswer.substring(3);
        }
        if (rawAnswer.endsWith("```")) {
            rawAnswer = rawAnswer.substring(0, rawAnswer.length - 3);
        }

        const quizData = JSON.parse(rawAnswer.trim());

        console.log(`Successfully generated ${quizData.length} quiz questions.`);
        return {
            quiz: quizData,
            sources: response.context.map((doc, i) => ({
                id: i + 1,
                page: doc.metadata.loc?.pageNumber || 'Unknown',
                snippet: doc.pageContent.substring(0, 100) + '...'
            }))
        };
    } catch (e) {
        console.error("Failed to parse the LLM response as JSON:", e);
        console.error("Raw LLM output was:", response.answer);
        throw new Error("Failed to generate a valid quiz. The AI did not return proper JSON.");
    }
}
