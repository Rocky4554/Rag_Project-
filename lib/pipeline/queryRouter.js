import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { createLLMWithFallback } from "../llm.js";
import { queryRAG, queryRAGStream, retrieveForQuery, responseText } from "./rag.js";
import { routerLog } from "../logger.js";

/**
 * Build a short document description from session metadata.
 * Gives the classifier enough context to route without any embedding cost.
 */
function buildDocumentContext(session) {
    const parts = [];

    if (session.originalName) {
        parts.push(`Document: "${session.originalName}"`);
    }

    // Use first few chunk snippets if available (fresh sessions only)
    if (session.docs && session.docs.length > 0) {
        const snippets = session.docs
            .slice(0, 3)
            .map(doc => doc.pageContent.substring(0, 200))
            .join("\n");
        parts.push(`Content preview:\n${snippets}`);
    }

    return parts.join("\n") || "Unknown document";
}

/**
 * Format recent chat history for the classifier (last 2 pairs).
 */
function formatRecentHistory(chatHistory) {
    if (!chatHistory || chatHistory.length === 0) return "";
    const recent = chatHistory.slice(-4);
    return recent.map(msg => {
        const role = msg.getType() === 'human' ? 'User' : 'Assistant';
        return `${role}: ${msg.content.substring(0, 150)}`;
    }).join('\n');
}

const ROUTER_SYSTEM_PROMPT = `You are a query router for a document Q&A system. Classify the user's message.

The user has uploaded:
{documentContext}

Recent conversation:
{chatHistory}

**DIRECT** — answer from general knowledge. Use this for:
- Greetings (Hello, Hi, Thanks, How are you)
- General knowledge questions NOT specifically about the uploaded document's content (e.g. "What is React?" when the document is about Python)
- Meta questions about you (What can you do? Who are you?)
- Simple follow-up acknowledgments (OK, Got it, I see)
- Questions on topics unrelated to the document

**RAG** — route to document search. Use this ONLY for:
- Questions that specifically need information from the uploaded document
- Requests to explain, summarize, or elaborate on document content
- Follow-up questions that reference previous document-based answers

Respond with ONLY the single word DIRECT or RAG.`;

const DIRECT_SYSTEM_PROMPT = `You are a helpful assistant on a document-based learning platform.

The user has uploaded:
{documentContext}

User's name: {userName}

- Answer the message FULLY and helpfully using your own knowledge. Give detailed, useful answers.
- Do NOT refuse or give short dismissals.
- When greeting, address the user by their name if provided (e.g. "Hello, Raunak!").
- After your answer, briefly mention you can also help with the uploaded document.`;

/** Returns "DIRECT" or "RAG"; any classifier failure falls back to RAG (safe default). */
async function classify(session, query, chatHistory) {
    const start = performance.now();
    const llm = createLLMWithFallback({ provider: "google", temperature: 0, maxTokens: 10 });
    try {
        const prompt = ROUTER_SYSTEM_PROMPT
            .replace("{documentContext}", buildDocumentContext(session))
            .replace("{chatHistory}", formatRecentHistory(chatHistory) || "(no prior messages)");
        const response = await llm.invoke([new SystemMessage(prompt), new HumanMessage(query)]);
        const route = /\bDIRECT\b/i.test(responseText(response)) ? "DIRECT" : "RAG";
        routerLog.info({ classifyMs: Math.round(performance.now() - start), route, query: query.substring(0, 80) }, 'Query routed');
        return route;
    } catch (err) {
        routerLog.warn({ ms: Math.round(performance.now() - start), err: err.message }, 'Classifier failed, falling back to RAG');
        return "RAG";
    }
}

function directMessages(session, query, chatHistory, userName) {
    const prompt = DIRECT_SYSTEM_PROMPT
        .replace("{documentContext}", buildDocumentContext(session))
        .replace("{userName}", userName || "unknown");
    return [new SystemMessage(prompt), ...chatHistory, new HumanMessage(query)];
}

/**
 * Classify and retrieve concurrently, so a RAG answer isn't delayed by the
 * classifier round-trip. Retrieval errors are held until we know RAG needs them.
 */
function classifyAndRetrieve(session, query, chatHistory) {
    return Promise.all([
        classify(session, query, chatHistory),
        retrieveForQuery(session.vectorStore, session.bm25Index, query, chatHistory)
            .then(docs => ({ docs }), error => ({ error })),
    ]);
}

/**
 * Classify a query and either answer directly or answer from the document.
 * @returns {{ route: "DIRECT"|"RAG", answer: string }}
 */
export async function routeQuery(session, query, chatHistory = [], userName = null) {
    routerLog.info({ query: query.substring(0, 80), userName: !!userName }, 'Query router classifying');
    const [route, retrieval] = await classifyAndRetrieve(session, query, chatHistory);

    if (route === "DIRECT") {
        const llm = createLLMWithFallback({ provider: "google", temperature: 0.3 });
        const answer = responseText(await llm.invoke(directMessages(session, query, chatHistory, userName))).trim();
        return { route, answer };
    }

    if (retrieval.error) throw retrieval.error;
    const answer = await queryRAG(session.vectorStore, session.bm25Index, query, chatHistory, { retrievedDocs: retrieval.docs });
    return { route, answer };
}

/**
 * Streaming version — yields tokens from either the direct answer or the RAG stream.
 */
export async function* routeQueryStream(session, query, chatHistory = [], userName = null) {
    routerLog.info({ query: query.substring(0, 80), streaming: true, userName: !!userName }, 'Query router stream classifying');
    const [route, retrieval] = await classifyAndRetrieve(session, query, chatHistory);

    if (route === "DIRECT") {
        const llm = createLLMWithFallback({ provider: "google", temperature: 0.3, streaming: true });
        const stream = await llm.stream(directMessages(session, query, chatHistory, userName));
        for await (const chunk of stream) {
            const token = responseText(chunk);
            if (token) yield token;
        }
        return;
    }

    if (retrieval.error) throw retrieval.error;
    yield* queryRAGStream(session.vectorStore, session.bm25Index, query, chatHistory, { retrievedDocs: retrieval.docs });
}
