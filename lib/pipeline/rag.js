import { createLLMWithFallback, detectRateLimitError } from "../llm.js";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { hybridRetrieve } from "./hybridRetrieval.js";
import { chatLog } from "../logger.js";
import dotenv from 'dotenv';
dotenv.config();

const QA_SYSTEM_PROMPT = `You are a helpful, knowledgeable assistant for a document-based learning platform. You have access to the following document context:

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

Keep answers concise and conversational. Do not refuse to answer — always be helpful.`;

/** Text of an LLM/chain response or stream chunk, untrimmed — chunk-edge spaces matter when streaming. */
export function responseText(response) {
    if (!response) return "";
    if (typeof response === "string") return response;
    if (typeof response.answer === "string") return response.answer;
    if (typeof response.output_text === "string") return response.output_text;
    if (typeof response.text === "string") return response.text;
    if (typeof response.content === "string") return response.content;
    if (Array.isArray(response.content)) {
        return response.content.map(c => (typeof c === "string" ? c : c?.text || "")).join("");
    }
    return "";
}

function extractAnswerText(response) {
    return responseText(response).trim();
}

/** Prefix the last 3 human+ai pairs so semantic search resolves follow-ups ("what about its cost?"). */
function buildContextualQuery(query, chatHistory) {
    if (chatHistory.length === 0) return query;
    const historyText = chatHistory.slice(-6).map(msg => {
        const role = msg.getType() === 'human' ? 'User' : 'Assistant';
        return `${role}: ${msg.content}`;
    }).join('\n');
    return `${historyText}\nUser: ${query}`;
}

function buildQAChain(llm) {
    const qaPrompt = ChatPromptTemplate.fromMessages([
        ["system", QA_SYSTEM_PROMPT],
        new MessagesPlaceholder("chat_history"),
        ["human", "{input}"]
    ]);
    return createStuffDocumentsChain({ llm, prompt: qaPrompt });
}

/**
 * Hybrid retrieval (BM25 + semantic → RRF merge → rerank → top 10) for a chat query.
 * Exposed so the query router can run it concurrently with classification.
 */
export async function retrieveForQuery(vectorStore, bm25Index, query, chatHistory = []) {
    const retrievalStart = performance.now();
    const docs = await hybridRetrieve(vectorStore, bm25Index, buildContextualQuery(query, chatHistory), query, { topK: 10 });
    const retrievalMs = Math.round(performance.now() - retrievalStart);
    const sources = docs.map((doc, i) => ({
        index: i + 1,
        page: doc.metadata.loc?.pageNumber || 'Unknown',
        snippet: doc.pageContent.substring(0, 80)
    }));
    chatLog.info({ retrievalMs, docsRetrieved: docs.length, sources }, 'RAG retrieval complete');
    return docs;
}

function logLLMError(err, llmMs, extra, label) {
    const rateLimitType = detectRateLimitError(err);
    if (rateLimitType) {
        chatLog.error({ rateLimitType, provider: 'google', llmMs, err: err.message, ...extra }, `${label} rate limit / quota error`);
    } else {
        chatLog.error({ llmMs, err: err.message, ...extra }, `${label} LLM call failed`);
    }
}

/**
 * @param {object} [opts]
 * @param {Array}  [opts.retrievedDocs] - docs already retrieved (skips retrieval)
 */
export async function queryRAG(vectorStore, bm25Index, query, chatHistory = [], { retrievedDocs } = {}) {
    const ragStart = performance.now();

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    chatLog.info({ query: query.substring(0, 120), historyLength: chatHistory.length }, 'RAG query started');

    const llm = createLLMWithFallback({ provider: "google", temperature: 0 });
    const combineDocsChain = await buildQAChain(llm);
    const docs = retrievedDocs || await retrieveForQuery(vectorStore, bm25Index, query, chatHistory);

    const llmStart = performance.now();
    let response;
    try {
        response = await combineDocsChain.invoke({
            context: docs,
            chat_history: chatHistory,
            input: query,
        });
    } catch (err) {
        logLLMError(err, Math.round(performance.now() - llmStart), {}, 'RAG');
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
        { totalMs, llmMs, answerLength: finalAnswer.length, query: query.substring(0, 80) },
        'RAG query complete'
    );

    return finalAnswer;
}

/**
 * Streaming version of queryRAG — yields tokens as they arrive from the LLM.
 */
export async function* queryRAGStream(vectorStore, bm25Index, query, chatHistory = [], { retrievedDocs } = {}) {
    const ragStart = performance.now();

    if (!process.env.GEMINI_API_KEY) {
        throw new Error("Missing GEMINI_API_KEY in environment variables");
    }

    chatLog.info({ query: query.substring(0, 120), historyLength: chatHistory.length, streaming: true }, 'RAG stream started');

    const llm = createLLMWithFallback({ provider: "google", temperature: 0, streaming: true });
    const combineDocsChain = await buildQAChain(llm);
    const docs = retrievedDocs || await retrieveForQuery(vectorStore, bm25Index, query, chatHistory);

    const llmStart = performance.now();
    let tokenCount = 0;
    let totalChars = 0;

    try {
        const stream = await combineDocsChain.stream({
            context: docs,
            chat_history: chatHistory,
            input: query,
        });

        for await (const chunk of stream) {
            const token = responseText(chunk);
            if (token) {
                tokenCount++;
                totalChars += token.length;
                yield token;
            }
        }
    } catch (err) {
        logLLMError(err, Math.round(performance.now() - llmStart), { tokenCount }, 'RAG stream');
        throw err;
    }

    const llmMs = Math.round(performance.now() - llmStart);
    const totalMs = Math.round(performance.now() - ragStart);

    chatLog.info(
        { totalMs, llmMs, tokenChunks: tokenCount, totalChars, query: query.substring(0, 80) },
        'RAG stream complete'
    );
}
