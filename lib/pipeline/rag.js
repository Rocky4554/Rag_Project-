import { createLLMWithFallback, detectRateLimitError } from "../llm.js";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { chatLog } from "../logger.js";
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
    const ragStart = performance.now();

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    chatLog.info({ query: query.substring(0, 120), historyLength: chatHistory.length }, 'RAG query started');

    // 1. Initialize the Chat Model with Fallback
    const llm = createLLMWithFallback({
        provider: "google",
        temperature: 0,
    });

    // 2. Manually build a context-aware query for the retriever
    let contextualQuery = query;
    if (chatHistory.length > 0) {
        const recentHistory = chatHistory.slice(-6); // last 3 human+ai pairs
        const historyText = recentHistory.map(msg => {
            const role = msg.getType() === 'human' ? 'User' : 'Assistant';
            return `${role}: ${msg.content}`;
        }).join('\n');
        contextualQuery = `${historyText}\nUser: ${query}`;
    }

    // 3. Define the main QA prompt
    const qaPrompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            `You are a helpful, knowledgeable assistant for a document-based learning platform. You have access to the following document context:

Context:
{context}

Follow these rules based on the type of message:

1. **Greetings & casual messages** (e.g. "Hello", "Thanks", "How are you"):
   Respond naturally and warmly. Briefly mention what the document is about so the user knows what they can ask.

2. **Questions answered by the document**:
   Answer using the document context. Cite specific details from it.

3. **Questions related to the document topic but not fully covered**:
   Use the document as a foundation, then supplement with your own knowledge. Clearly distinguish what comes from the document vs your general knowledge by saying something like "Based on the document..." and "Beyond what the document covers...".

4. **Questions completely unrelated to the document**:
   Briefly answer if it's a simple/quick question, then gently guide the user back by saying something like "By the way, I'm best at helping with questions about [document topic]. Feel free to ask me anything about that!"

Keep answers concise and conversational. Do not refuse to answer — always be helpful.`
        ],
        new MessagesPlaceholder("chat_history"),
        ["human", "{input}"]
    ]);

    // 4. Build retriever + combine chain.
    const retriever = vectorStore.asRetriever({ k: 3 });

    const combineDocsChain = await createStuffDocumentsChain({
        llm: llm,
        prompt: qaPrompt,
    });

    // 5. Retrieve docs
    const retrievalStart = performance.now();
    const retrievedDocs = await retriever.invoke(contextualQuery);
    const retrievalMs = Math.round(performance.now() - retrievalStart);

    const sources = retrievedDocs.map((doc, i) => ({
        index: i + 1,
        page: doc.metadata.loc?.pageNumber || 'Unknown',
        snippet: doc.pageContent.substring(0, 80)
    }));
    chatLog.info({ retrievalMs, docsRetrieved: retrievedDocs.length, sources }, 'RAG retrieval complete');

    // 6. LLM call
    const llmStart = performance.now();
    let response;
    try {
        response = await combineDocsChain.invoke({
            context: retrievedDocs,
            chat_history: chatHistory,
            input: query,
        });
    } catch (err) {
        const llmMs = Math.round(performance.now() - llmStart);
        const rateLimitType = detectRateLimitError(err);
        if (rateLimitType) {
            chatLog.error({ rateLimitType, provider: 'google', llmMs, err: err.message }, 'RAG LLM rate limit / quota error');
        } else {
            chatLog.error({ llmMs, err: err.message }, 'RAG LLM call failed');
        }
        throw err;
    }
    const llmMs = Math.round(performance.now() - llmStart);

    const finalAnswer = extractAnswerText(response);
    const totalMs = Math.round(performance.now() - ragStart);

    if (!finalAnswer) {
        chatLog.warn({ totalMs, llmMs }, 'RAG produced empty answer, returning fallback');
        return "I don't know based on the provided document.";
    }

    chatLog.info(
        { totalMs, retrievalMs, llmMs, answerLength: finalAnswer.length, query: query.substring(0, 80) },
        'RAG query complete'
    );

    return finalAnswer;
}

/**
 * Streaming version of queryRAG — yields tokens as they arrive from the LLM.
 */
export async function* queryRAGStream(vectorStore, query, chatHistory = []) {
    const ragStart = performance.now();

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    chatLog.info({ query: query.substring(0, 120), historyLength: chatHistory.length, streaming: true }, 'RAG stream started');

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
            `You are a helpful, knowledgeable assistant for a document-based learning platform. You have access to the following document context:

Context:
{context}

Follow these rules based on the type of message:

1. **Greetings & casual messages** (e.g. "Hello", "Thanks", "How are you"):
   Respond naturally and warmly. Briefly mention what the document is about so the user knows what they can ask.

2. **Questions answered by the document**:
   Answer using the document context. Cite specific details from it.

3. **Questions related to the document topic but not fully covered**:
   Use the document as a foundation, then supplement with your own knowledge. Clearly distinguish what comes from the document vs your general knowledge by saying something like "Based on the document..." and "Beyond what the document covers...".

4. **Questions completely unrelated to the document**:
   Briefly answer if it's a simple/quick question, then gently guide the user back by saying something like "By the way, I'm best at helping with questions about [document topic]. Feel free to ask me anything about that!"

Keep answers concise and conversational. Do not refuse to answer — always be helpful.`
        ],
        new MessagesPlaceholder("chat_history"),
        ["human", "{input}"]
    ]);

    const retriever = vectorStore.asRetriever({ k: 3 });

    const combineDocsChain = await createStuffDocumentsChain({
        llm,
        prompt: qaPrompt,
    });

    const retrievalStart = performance.now();
    const retrievedDocs = await retriever.invoke(contextualQuery);
    const retrievalMs = Math.round(performance.now() - retrievalStart);
    chatLog.info({ retrievalMs, docsRetrieved: retrievedDocs.length }, 'RAG stream retrieval complete');

    const llmStart = performance.now();
    let tokenCount = 0;
    let totalChars = 0;

    try {
        const stream = await combineDocsChain.stream({
            context: retrievedDocs,
            chat_history: chatHistory,
            input: query,
        });

        for await (const chunk of stream) {
            const token = extractAnswerText(chunk);
            if (token) {
                tokenCount++;
                totalChars += token.length;
                yield token;
            }
        }
    } catch (err) {
        const llmMs = Math.round(performance.now() - llmStart);
        const rateLimitType = detectRateLimitError(err);
        if (rateLimitType) {
            chatLog.error({ rateLimitType, provider: 'google', llmMs, err: err.message }, 'RAG stream rate limit / quota error');
        } else {
            chatLog.error({ llmMs, tokenCount, err: err.message }, 'RAG stream LLM error');
        }
        throw err;
    }

    const llmMs = Math.round(performance.now() - llmStart);
    const totalMs = Math.round(performance.now() - ragStart);

    chatLog.info(
        { totalMs, retrievalMs, llmMs, tokenChunks: tokenCount, totalChars, query: query.substring(0, 80) },
        'RAG stream complete'
    );
}
