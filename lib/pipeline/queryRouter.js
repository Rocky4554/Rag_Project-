import { createLLMWithFallback } from "../llm.js";
import { queryRAG, queryRAGStream } from "./rag.js";
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

const ROUTER_SYSTEM_PROMPT = `You are a query router for a document Q&A system. Classify the user's message and either respond directly OR route it to the document search system.

The user has uploaded:
{documentContext}

User's name: {userName}

Recent conversation:
{chatHistory}

Choose one of two routes:

**DIRECT** — Respond using your own knowledge. Use this for:
- Greetings (Hello, Hi, Thanks, How are you)
- General knowledge questions NOT specifically about the uploaded document's content (e.g. "What is React?" when the document is about Python)
- Meta questions about you (What can you do? Who are you?)
- Simple follow-up acknowledgments (OK, Got it, I see)
- Questions on topics unrelated to the document

When responding DIRECT:
- Answer the question FULLY and helpfully using your own knowledge. Give detailed, useful answers.
- Do NOT refuse or give short dismissals.
- When greeting, address the user by their name if provided (e.g. "Hello, Raunak!").
- After your answer, briefly mention you can also help with the uploaded document.

**RAG** — Route to document search. Use this ONLY for:
- Questions that specifically need information from the uploaded document
- Requests to explain, summarize, or elaborate on document content
- Follow-up questions that reference previous document-based answers

Respond ONLY with valid JSON (no markdown fences, no backticks):
{"route": "DIRECT", "response": "your full helpful answer here"}
or
{"route": "RAG", "refinedQuery": "optional clearer version of the query for document search"}`;

/**
 * Classify a query and either respond directly or delegate to RAG.
 * @returns {{ route: "DIRECT"|"RAG", answer: string }}
 */
export async function routeQuery(session, query, chatHistory = [], userName = null) {
    const start = performance.now();

    const documentContext = buildDocumentContext(session);
    const historyText = formatRecentHistory(chatHistory);

    routerLog.info({ query: query.substring(0, 80), userName: !!userName }, 'Query router classifying');

    const classifierLLM = createLLMWithFallback({
        provider: "google",
        temperature: 0.3,
    });

    let classification;
    try {
        const prompt = ROUTER_SYSTEM_PROMPT
            .replace("{documentContext}", documentContext)
            .replace("{userName}", userName || "unknown")
            .replace("{chatHistory}", historyText || "(no prior messages)");

        const response = await classifierLLM.invoke([
            { role: "system", content: prompt },
            { role: "user", content: query },
        ]);

        const text = typeof response.content === 'string'
            ? response.content
            : response.content.map(c => c.text || '').join('');

        // Strip accidental markdown fences
        const cleaned = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        classification = JSON.parse(cleaned);
    } catch (err) {
        // Classifier failed — fall through to RAG (safe default)
        const ms = Math.round(performance.now() - start);
        routerLog.warn({ ms, err: err.message }, 'Classifier failed, falling back to RAG');

        const answer = await queryRAG(session.vectorStore, query, chatHistory);
        return { route: "RAG", answer };
    }

    const classifyMs = Math.round(performance.now() - start);

    if (classification.route === "DIRECT" && classification.response) {
        routerLog.info({ classifyMs, route: 'DIRECT', query: query.substring(0, 80) }, 'Query routed directly');
        return { route: "DIRECT", answer: classification.response };
    }

    // Route to RAG
    const effectiveQuery = classification.refinedQuery || query;
    routerLog.info({ classifyMs, route: 'RAG', refined: !!classification.refinedQuery }, 'Query routed to RAG');

    const answer = await queryRAG(session.vectorStore, effectiveQuery, chatHistory);
    return { route: "RAG", answer };
}

/**
 * Streaming version — yields tokens from either direct response or RAG stream.
 */
export async function* routeQueryStream(session, query, chatHistory = [], userName = null) {
    const start = performance.now();

    const documentContext = buildDocumentContext(session);
    const historyText = formatRecentHistory(chatHistory);

    routerLog.info({ query: query.substring(0, 80), streaming: true, userName: !!userName }, 'Query router stream classifying');

    const classifierLLM = createLLMWithFallback({
        provider: "google",
        temperature: 0.3,
    });

    let classification;
    try {
        const prompt = ROUTER_SYSTEM_PROMPT
            .replace("{documentContext}", documentContext)
            .replace("{userName}", userName || "unknown")
            .replace("{chatHistory}", historyText || "(no prior messages)");

        const response = await classifierLLM.invoke([
            { role: "system", content: prompt },
            { role: "user", content: query },
        ]);

        const text = typeof response.content === 'string'
            ? response.content
            : response.content.map(c => c.text || '').join('');

        const cleaned = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        classification = JSON.parse(cleaned);
    } catch (err) {
        const ms = Math.round(performance.now() - start);
        routerLog.warn({ ms, err: err.message, streaming: true }, 'Classifier failed, falling back to RAG stream');

        yield* queryRAGStream(session.vectorStore, query, chatHistory);
        return;
    }

    const classifyMs = Math.round(performance.now() - start);

    if (classification.route === "DIRECT" && classification.response) {
        routerLog.info({ classifyMs, route: 'DIRECT', streaming: true }, 'Query stream routed directly');

        // Yield the response in word-sized chunks to simulate streaming
        const words = classification.response.split(/(\s+)/);
        for (const word of words) {
            if (word) yield word;
        }
        return;
    }

    routerLog.info({ classifyMs, route: 'RAG', streaming: true }, 'Query stream routed to RAG');
    const effectiveQuery = classification.refinedQuery || query;
    yield* queryRAGStream(session.vectorStore, effectiveQuery, chatHistory);
}
