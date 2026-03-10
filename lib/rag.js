import { createLLMWithFallback } from "./llm.js";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import dotenv from 'dotenv';
dotenv.config();

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

    // 4. Create the chains using the standard retriever (no history-aware wrapper needed)
    const retriever = vectorStore.asRetriever({ k: 3 });

    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: qaPrompt,
    });

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: combineDocsChain,
        retriever: retriever,
    });

    // 5. Invoke — pass the original query as {input} for the LLM,
    //    but use contextualQuery for the retriever search step
    console.log("Generating answer using LangChain and Gemini...");
    const response = await retrievalChain.invoke({
        chat_history: chatHistory,
        input: contextualQuery,   // retriever uses this to search ChromaDB
    });

    console.log("\n--- Sources Used ---");
    response.context.forEach((doc, i) => {
        console.log(`Source ${i + 1} (Page ${doc.metadata.loc?.pageNumber || 'Unknown'}): ${doc.pageContent.substring(0, 50)}...`);
    });
    console.log("--------------------\n");

    return response.answer;
}