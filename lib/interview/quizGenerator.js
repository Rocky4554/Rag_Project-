import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { quizLog } from "../logger.js";
import { detectRateLimitError } from "../llm.js";
import dotenv from 'dotenv';
dotenv.config();

export async function generateQuiz(vectorStore, options = {}) {
    const quizStart = performance.now();

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    const {
        topic = "general concepts",
        numQuestions = 5
    } = options;

    quizLog.info({ topic, numQuestions }, 'Quiz generation started');

    // 1. Initialize the Chat Model (Gemini)
    const llm = new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: "gemini-3-flash-preview",
        temperature: 0.2,
        maxOutputTokens: 8192,
    });

    // 2. Define the System Prompt using LangChain Templates
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

    const retriever = vectorStore.asRetriever({
        k: 10,
    });

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: combineDocsChain,
        retriever: retriever,
    });

    // 4. Invoke the chain
    const searchQuery = topic && topic.toLowerCase() !== "general" && topic.trim() !== ""
        ? `information about ${topic} suitable for a quiz`
        : `key concepts, definitions, main ideas, and important facts`;

    quizLog.info({ searchQuery, model: 'gemini-3-flash-preview' }, 'Quiz LLM call started');
    const llmStart = performance.now();

    let response;
    try {
        response = await retrievalChain.invoke({
            input: searchQuery,
            topic: topic || "general concepts",
            numQuestions: numQuestions.toString()
        });
    } catch (err) {
        const llmMs = Math.round(performance.now() - llmStart);
        const rateLimitType = detectRateLimitError(err);
        if (rateLimitType) {
            quizLog.error({ rateLimitType, provider: 'google', model: 'gemini-3-flash-preview', llmMs, err: err.message }, 'Quiz LLM rate limit / quota error');
        } else {
            quizLog.error({ llmMs, err: err.message }, 'Quiz LLM call failed');
        }
        throw err;
    }

    const llmMs = Math.round(performance.now() - llmStart);
    quizLog.info({ llmMs, answerLength: response.answer?.length || 0, contextDocs: response.context?.length || 0 }, 'Quiz LLM response received');

    // 5. Parse the LLM output (which should be JSON)
    try {
        let rawAnswer = response.answer;
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
        const totalMs = Math.round(performance.now() - quizStart);

        quizLog.info(
            { totalMs, llmMs, questionsGenerated: quizData.length, topic, numQuestions },
            'Quiz generation complete'
        );

        return {
            quiz: quizData,
            sources: response.context.map((doc, i) => ({
                id: i + 1,
                page: doc.metadata.loc?.pageNumber || 'Unknown',
                snippet: doc.pageContent.substring(0, 100) + '...'
            }))
        };
    } catch (e) {
        const totalMs = Math.round(performance.now() - quizStart);
        quizLog.error(
            { totalMs, llmMs, err: e.message, rawOutput: response.answer?.substring(0, 200) },
            'Quiz JSON parse failed'
        );
        throw new Error("Failed to generate a valid quiz. The AI did not return proper JSON.");
    }
}
