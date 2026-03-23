import { createLLMWithFallback } from "../llm.js";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import dotenv from 'dotenv';
dotenv.config();

function extractAnswerText(response) {
    if (!response) return "";
    if (typeof response === "string") return response.trim();
    if (typeof response.answer === "string") return response.answer.trim();
    if (typeof response.output_text === "string") return response.output_text.trim();
    if (typeof response.text === "string") return response.text.trim();
    if (typeof response.content === "string") return response.content.trim();
    if (Array.isArray(response.content)) {
        return response.content.map(c => (typeof c === "string" ? c : c?.text || "")).join("").trim();
    }
    return "";
}

export async function queryRAG(vectorStore, query, chatHistory = []) {
    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    console.log(`Setting up LangChain RAG pipeline for: "${query}"`);

    // 1. Initialize the Chat Model with Fallback
    const llm = createLLMWithFallback({
        provider: "google",
        temperature: 0,
    });

    // 2. Manually build a context-aware query for the retriever
    // By prepending the last 3 turns of conversation, the vector search
    // can understand follow-up questions like "what does IT do?" correctly.
    let contextualQuery = query;
    if (chatHistory.length > 0) {
        const recentHistory = chatHistory.slice(-6); // last 3 human+ai pairs
        const historyText = recentHistory.map(msg => {
            const role = msg.getType() === 'human' ? 'User' : 'Assistant';
            return `${role}: ${msg.content}`;
        }).join('\n');
        contextualQuery = `${historyText}\nUser: ${query}`;
    }

    // 3. Define the main QA prompt — history is passed here so the LLM answers with context
    const qaPrompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are a helpful assistant answering a question based strictly on the provided context.
If the answer is not contained in the context, say "I don't know based on the provided document".

Context:
{context}`
        ],
        new MessagesPlaceholder("chat_history"),
        ["human", "{input}"]
    ]);

    // 4. Build retriever + combine chain.
    // We retrieve using contextualQuery but answer using the original user question.
    const retriever = vectorStore.asRetriever({ k: 3 });

    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: qaPrompt,
    });

    // 5. Retrieve docs with contextual query, then ask LLM using original query.
    console.log("Generating answer using LangChain and Gemini...");
    const retrievedDocs = await retriever.invoke(contextualQuery);
    const response = await combineDocsChain.invoke({
        context: retrievedDocs,
        chat_history: chatHistory,
        input: query,
    });

    console.log("\n--- Sources Used ---");
    retrievedDocs.forEach((doc, i) => {
        console.log(`Source ${i + 1} (Page ${doc.metadata.loc?.pageNumber || 'Unknown'}): ${doc.pageContent.substring(0, 50)}...`);
    });
    console.log("--------------------\n");

    const finalAnswer = extractAnswerText(response);
    if (!finalAnswer) {
        console.warn("[RAG] Empty answer generated. Returning fallback message.");
        return "I don't know based on the provided document.";
    }
    return finalAnswer;
}

/**
 * Streaming version of queryRAG — yields tokens as they arrive from the LLM.
 */
export async function* queryRAGStream(vectorStore, query, chatHistory = []) {
    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    const llm = createLLMWithFallback({
        provider: "google",
        temperature: 0,
        streaming: true,
    });

    let contextualQuery = query;
    if (chatHistory.length > 0) {
        const recentHistory = chatHistory.slice(-6);
        const historyText = recentHistory.map(msg => {
            const role = msg.getType() === 'human' ? 'User' : 'Assistant';
            return `${role}: ${msg.content}`;
        }).join('\n');
        contextualQuery = `${historyText}\nUser: ${query}`;
    }

    const qaPrompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are a helpful assistant answering a question based strictly on the provided context.
If the answer is not contained in the context, say "I don't know based on the provided document".

Context:
{context}`
        ],
        new MessagesPlaceholder("chat_history"),
        ["human", "{input}"]
    ]);

    const retriever = vectorStore.asRetriever({ k: 3 });

    const combineDocsChain = await createStuffDocumentsChain({
        llm,
        prompt: qaPrompt,
    });

    const retrievedDocs = await retriever.invoke(contextualQuery);

    const stream = await combineDocsChain.stream({
        context: retrievedDocs,
        chat_history: chatHistory,
        input: query,
    });

    for await (const chunk of stream) {
        const token = extractAnswerText(chunk);
        if (token) yield token;
    }
}