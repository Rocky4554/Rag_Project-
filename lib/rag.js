import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';
dotenv.config();

export async function queryRAG(vectorStore, query) {
    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    console.log(`Setting up LangChain RAG pipeline for: "${query}"`);

    // 1. Initialize the Chat Model (Gemini)
    const llm = new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: "gemini-2.5-flash",
        temperature: 0, // Keep temperature low for factual RAG responses
    });

    // 2. Define the System Prompt using LangChain Templates
    // LangChain will automatically inject the retrieved text into the {context} variable
    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are a helpful assistant answering a question based strictly on the provided context.
If the answer is not contained in the context, say "I don't know based on the provided document".

Context:
{context}`
        ],
        ["human", "{input}"]
    ]);

    // 3. Create the chains
    // The "Stuff Documents" chain simply stuffs all retrieved docs into the {context} prompt variable
    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: prompt,
    });

    // The retrieval chain uses the vector store to fetch relevant docs, then passes them to the combine chain
    const retriever = vectorStore.asRetriever({
        k: 3, // Number of documents to retrieve
    });

    const retrievalChain = await createRetrievalChain({
        combineDocsChain: combineDocsChain,
        retriever: retriever,
    });

    // 4. Invoke the chain!
    console.log("Generating answer using LangChain and Gemini...");
    const response = await retrievalChain.invoke({
        input: query,
    });

    // The response object contains both the final string answer AND the source documents it used
    console.log("\n--- Sources Used ---");
    response.context.forEach((doc, i) => {
        console.log(`Source ${i + 1} (Page ${doc.metadata.loc?.pageNumber || 'Unknown'}): ${doc.pageContent.substring(0, 50)}...`);
    });
    console.log("--------------------\n");

    return response.answer;
}