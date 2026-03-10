import { StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
import { ChatGroq } from "@langchain/groq";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import fs from "fs";
import dotenv from "dotenv";
dotenv.config();

// ═══════════════════════════════════════════════════════════════════
// LLM LOGGER
// ═══════════════════════════════════════════════════════════════════
const globalLLMLogger = {
    runs: new Map(),
    handleLLMStart(llm, prompts, runId) {
        this.runs.set(runId, Date.now());
        console.log(`\n[LLM] 🟢 Request started -> Provider: Groq, Model: llama-3.3-70b-versatile`);
    },
    handleLLMEnd(output, runId) {
        const duration = Date.now() - (this.runs.get(runId) || Date.now());
        this.runs.delete(runId);

        let pTokens = 0, cTokens = 0, tTokens = 0;
        if (output.llmOutput?.tokenUsage) {
            pTokens = output.llmOutput.tokenUsage.promptTokens || 0;
            cTokens = output.llmOutput.tokenUsage.completionTokens || 0;
            tTokens = output.llmOutput.tokenUsage.totalTokens || 0;
        } else if (output.generations?.[0]?.[0]?.message?.response_metadata?.tokenUsage) {
            const usage = output.generations[0][0].message.response_metadata.tokenUsage;
            pTokens = usage.promptTokens || 0;
            cTokens = usage.completionTokens || 0;
            tTokens = usage.totalTokens || 0;
        }

        console.log(`[LLM] 🔴 Response received in ${duration}ms`);
        console.log(`[LLM] 📊 Tokens -> Prompt: ${pTokens} | Completion: ${cTokens} | Total: ${tTokens}\n`);
    },
    handleLLMError(error, runId) {
        const duration = Date.now() - (this.runs.get(runId) || Date.now());
        this.runs.delete(runId);
        console.log(`[LLM] ❌ Error after ${duration}ms: ${error?.message || "Unknown error"}\n`);
    }
};

// ═══════════════════════════════════════════════════════════════════
// TTS UTILITIES
// ═══════════════════════════════════════════════════════════════════

export function stripMarkdownForTTS(text) {
    return text
        .replace(/#{1,6}\s+/g, "")
        .replace(/\*\*(.+?)\*\*/g, "$1")
        .replace(/\*(.+?)\*/g, "$1")
        .replace(/`(.+?)`/g, "$1")
        .replace(/```[\s\S]*?```/g, "")
        .replace(/\[(.+?)\]\(.+?\)/g, "$1")
        .replace(/[-*•]\s+/g, "")
        .replace(/\n{2,}/g, " ")
        .replace(/\n/g, " ")
        .trim();
}

const TTS_PHRASE_CACHE = {
    "lets_move_on": "assets/tts/lets_move_on.mp3",
    "great_answer": "assets/tts/great_answer.mp3",
    "good_effort": "assets/tts/good_effort.mp3",
    "take_your_time": "assets/tts/take_your_time.mp3",
    "no_worries": "assets/tts/no_worries.mp3",
    "interesting": "assets/tts/interesting.mp3",
    "next_question": "assets/tts/next_question.mp3",
    "final_question": "assets/tts/final_question.mp3",
    "thats_okay": "assets/tts/thats_okay.mp3",
    "thanks_for_time": "assets/tts/thanks_for_time.mp3",
    "interview_intro": "assets/tts/interview_intro.mp3",
    "interview_outro": "assets/tts/interview_outro.mp3",
    "interview_stopped": "assets/tts/interview_stopped.mp3",
    "out_of_context": "assets/tts/out_of_context.mp3",
};

export function checkTTSCache(phraseKey) {
    const filePath = TTS_PHRASE_CACHE[phraseKey];
    if (filePath && fs.existsSync(filePath)) return filePath;
    return null;
}

export function parseTTSResponse(responseText) {
    const match = responseText.match(/\[(\w+)\]/);
    if (match) {
        const phraseKey = match[1];
        const uniquePart = responseText.replace(/\[(\w+)\]/, "").trim();
        return {
            phraseKey: TTS_PHRASE_CACHE[phraseKey] ? phraseKey : null,
            uniquePart: stripMarkdownForTTS(uniquePart),
        };
    }
    return { phraseKey: null, uniquePart: stripMarkdownForTTS(responseText) };
}

// ═══════════════════════════════════════════════════════════════════
// VECTORSTORE REGISTRY
// ═══════════════════════════════════════════════════════════════════
const vectorStoreRegistry = new Map();
const ragCache = new Map();

export function registerVectorStore(sessionId, vectorStore) {
    vectorStoreRegistry.set(sessionId, vectorStore);
}
function getVectorStore(sessionId) {
    const vs = vectorStoreRegistry.get(sessionId);
    if (!vs) throw new Error(`No vectorStore for session: ${sessionId}`);
    return vs;
}
async function getContext(sessionId, query, k = 3) {
    const key = `${sessionId}:${query.substring(0, 40)}`;
    if (ragCache.has(key)) return ragCache.get(key);
    const vs = getVectorStore(sessionId);
    const docs = await vs.asRetriever({ k }).invoke(query);
    const ctx = docs.map(d => d.pageContent).join("\n\n").substring(0, 700);
    ragCache.set(key, ctx);
    return ctx;
}

// ═══════════════════════════════════════════════════════════════════
// SOCKET REGISTRY — stores active socket per session for streaming
// ═══════════════════════════════════════════════════════════════════
const socketRegistry = new Map();

export function registerSocket(sessionId, socket) {
    socketRegistry.set(sessionId, socket);
}
export function unregisterSocket(sessionId) {
    socketRegistry.delete(sessionId);
}
function getSocket(sessionId) {
    return socketRegistry.get(sessionId) || null;
}

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════
export const interviewStateChannels = {
    sessionId: { value: (x, y) => y ?? x, default: () => "" },
    chatHistory: { value: (x, y) => y ?? x, default: () => [] },
    conversationSummary: { value: (x, y) => y ?? x, default: () => "" },   // ← NEW: compressed memory
    currentQuestion: { value: (x, y) => y ?? x, default: () => "" },
    userAnswer: { value: (x, y) => y ?? x, default: () => "" },
    evaluation: { value: (x, y) => y ?? x, default: () => null },
    difficultyLevel: { value: (x, y) => y ?? x, default: () => "medium" },
    questionsAsked: { value: (x, y) => y ?? x, default: () => 0 },
    maxQuestions: { value: (x, y) => y ?? x, default: () => 5 },
    topicsUsed: { value: (x, y) => y ?? x, default: () => [] },
    finalReport: { value: (x, y) => y ?? x, default: () => null },
    interviewStopped: { value: (x, y) => y ?? x, default: () => false },
    intent: { value: (x, y) => y ?? x, default: () => "normal" },
    scores: { value: (x, y) => y ?? x, default: () => [] },
    candidateName: { value: (x, y) => y ?? x, default: () => "there" },
    // ← NEW: pre-fetched context for next question (from parallel node)
    prefetchedContext: { value: (x, y) => y ?? x, default: () => "" },
};

const INTENTS = { STOP: "stop", SKIP: "skip", NERVOUS: "nervous", UNWELL: "unwell", IRRELEVANT: "irrelevant", NORMAL: "normal" };

// ═══════════════════════════════════════════════════════════════════
// LLM FACTORY
// ═══════════════════════════════════════════════════════════════════
function makeLLM(temperature = 0.7, maxTokens = 200, streaming = false) {
    return new ChatGroq({
        apiKey: process.env.GROQ_API_KEY,
        model: "llama-3.3-70b-versatile",
        temperature,
        maxTokens: maxTokens,
        streaming,
        callbacks: [globalLLMLogger],
    });
}

function buildCompactHistory(history, maxTurns = 3) {
    return history.slice(-maxTurns * 2).map(m => {
        const role = m instanceof HumanMessage ? "Candidate" : "Interviewer";
        const txt = m.content.length > 100 ? m.content.substring(0, 100) + "…" : m.content;
        return `${role}: ${txt}`;
    }).join("\n");
}

// ═══════════════════════════════════════════════════════════════════
// NODE 1 — DETECT INTENT
// ═══════════════════════════════════════════════════════════════════
async function detectIntentNode(state) {
    const llm = makeLLM(0.0, 10);
    const response = await llm.invoke([
        new SystemMessage(
            `You classify a candidate's interview response into one intent. Output exactly one word.
Speech-to-text may have typos. Be lenient with spelling (e.g., "saw the interview" means "stop", "dont no" means "skip").

INTENTS:
- stop   → wants to end/quit/stop the interview entirely
- skip   → doesn't know the answer, wants to pass or skip this question
- nervous → expressing anxiety, stress, blanking, freezing, fear
- unwell → feeling sick, tired, unwell, needs a break for health reasons
- irrelevant → candidate is talking about something completely unrelated to the technical question or interview context (out of context)
- normal → any genuine attempt to answer the question, even if wrong or short

OUTPUT: one word only. No punctuation. No explanation.`
        ),
        new HumanMessage(`Classify: "${state.userAnswer}"`)
    ]);

    const intent = response.content.trim().toLowerCase().replace(/[^a-z]/g, "");
    const validated = Object.values(INTENTS).includes(intent) ? intent : INTENTS.NORMAL;
    console.log(`[Intent] "${state.userAnswer.substring(0, 40)}…" → ${validated}`);
    return { intent: validated };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 2 — HANDLE EDGE CASES
// ═══════════════════════════════════════════════════════════════════
async function handleEdgeCaseNode(state) {
    const socket = getSocket(state.sessionId);

    // ── Fast path: known intents use cached audio ONLY. No LLM, no Polly.
    // ── The cached mp3 IS the full response — nothing else is generated.
    const CACHE_ONLY_INTENTS = {
        [INTENTS.STOP]: "interview_stopped",  // Full goodbye — plays cache + outro
        [INTENTS.SKIP]: "thats_okay",         // Full "that's okay, let's move on"
        [INTENTS.NERVOUS]: "take_your_time",     // Full calm-down message
        [INTENTS.UNWELL]: "no_worries",         // Full empathy + end interview
        [INTENTS.IRRELEVANT]: "out_of_context",    // Full redirect back to question
    };

    const cachedKey = CACHE_ONLY_INTENTS[state.intent];

    if (cachedKey) {
        // Purely cache-based: return the cached phrase key as the feedback text.
        // server.js will load the mp3 directly — no TTS API call needed.
        console.log(`[EdgeCase] Intent "${state.intent}" → using cache only: ${cachedKey}.mp3`);
        const responseText = cachedKey; // signal to server: "serve this cache key"
        const stopped = state.intent === INTENTS.STOP || state.intent === INTENTS.UNWELL;
        return {
            chatHistory: [...state.chatHistory,
            new HumanMessage(state.userAnswer),
            new AIMessage(`[${cachedKey}]`)
            ],
            interviewStopped: stopped,
            evaluation: {
                score: 0, accuracy: 0, clarity: 0, depth: 0,
                feedback: `[${cachedKey}]`,  // parseTTSResponse will extract the key
                nextDifficulty: "same"
            }
        };
    }

    // ── Fallback: unexpected / unplanned intent → LLM generates + Polly speaks ──
    console.log(`[EdgeCase] Intent "${state.intent}" is unexpected, falling back to LLM generation.`);
    const llm = makeLLM(0.85, 120, true);
    const prompt = `You are a professional AI interview assistant. The candidate said something unexpected during a technical interview: "${state.userAnswer}".
Write one natural, professional sentence acknowledging this and gently redirecting them back to the interview.`;

    let responseText = "";
    if (socket) {
        const stream = await llm.stream([
            new SystemMessage("You write natural spoken interviewer responses. No markdown, no lists. Sound human."),
            new HumanMessage(prompt)
        ]);
        for await (const chunk of stream) {
            const token = chunk.content;
            if (token) {
                responseText += token;
                socket.emit("ai_stream", { token, type: "edge_case" });
            }
        }
        socket.emit("ai_stream_end", { type: "edge_case" });
    } else {
        const response = await llm.invoke([
            new SystemMessage("You write natural spoken interviewer responses. No markdown, no lists. Sound human."),
            new HumanMessage(prompt)
        ]);
        responseText = response.content.trim();
    }

    return {
        chatHistory: [...state.chatHistory,
        new HumanMessage(state.userAnswer),
        new AIMessage(responseText)
        ],
        interviewStopped: false,
        evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: responseText, nextDifficulty: "same" }
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 3 — EVALUATE ANSWER  (runs in PARALLEL with prefetchContext)
// ═══════════════════════════════════════════════════════════════════
async function evaluateAnswerNode(state) {
    console.log(`[Interview] Evaluating Q#${state.questionsAsked}`);
    const context = await getContext(state.sessionId, state.currentQuestion);
    const socket = getSocket(state.sessionId);

    const systemPrompt =
        `You are a senior engineer conducting a technical interview. Evaluate the candidate's spoken answer.

QUESTION: ${state.currentQuestion}
CANDIDATE SAID: "${state.userAnswer}"
REFERENCE MATERIAL: ${context}

Scoring guide:
- score 1-3: Wrong or vague
- score 4-6: Partially correct
- score 7-8: Solid answer
- score 9-10: Complete and well-explained

Feedback rules:
- Write as if SPEAKING to the candidate
- Start feedback with exactly one tag:
  [great_answer] if score >= 8
  [good_effort]  if score <= 7
- After the tag write ONE specific, useful spoken sentence
- Sound like a thoughtful human, not a rubric

Return ONLY valid JSON. No markdown. No extra text.
{"score":7,"accuracy":6,"clarity":8,"depth":6,"feedback":"[good_effort] You nailed the concept but missed the sliding window detail.","nextDifficulty":"same"}

nextDifficulty = "easier" | "same" | "harder"`;

    // Evaluation needs exact JSON → no streaming (would fragment JSON)
    const llm = makeLLM(0.2, 150);
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Evaluate now.")
    ]);

    let evalData;
    try {
        const clean = response.content.trim().replace(/```json\n?|```/g, "");
        evalData = JSON.parse(clean);
    } catch {
        evalData = { score: 5, accuracy: 5, clarity: 5, depth: 5, feedback: "[good_effort] Got it, let's keep going.", nextDifficulty: "same" };
    }

    // Stream just the feedback text to the client immediately
    if (socket && evalData.feedback) {
        socket.emit("ai_feedback", { feedback: evalData.feedback, score: evalData.score });
    }

    return {
        evaluation: evalData,
        chatHistory: [...state.chatHistory,
        new HumanMessage(state.userAnswer),
        new AIMessage(evalData.feedback)
        ],
        scores: [...state.scores, evalData.score]
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 3b — PREFETCH CONTEXT  (runs in PARALLEL with evaluateAnswer)
// Pre-fetches the RAG context for the next question while evaluation runs.
// ═══════════════════════════════════════════════════════════════════
async function prefetchContextNode(state) {
    console.log("[Interview] Prefetching context for next question…");
    try {
        const ctx = await getContext(state.sessionId, "core concepts definitions key ideas", 3);
        return { prefetchedContext: ctx };
    } catch {
        return { prefetchedContext: "" };
    }
}

// ═══════════════════════════════════════════════════════════════════
// NODE 4 — ADAPT DIFFICULTY
// ═══════════════════════════════════════════════════════════════════
async function adaptDifficultyNode(state) {
    const levels = ["easy", "medium", "hard"];
    let idx = Math.max(0, levels.indexOf(state.difficultyLevel));
    if (state.evaluation?.nextDifficulty === "harder" && idx < 2) idx++;
    else if (state.evaluation?.nextDifficulty === "easier" && idx > 0) idx--;
    return { difficultyLevel: levels[idx] };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 4b — COMPRESS MEMORY (runs after every 2nd answered question)
// Summarizes full chat history into ~3 sentences. Keeps token usage low.
// ═══════════════════════════════════════════════════════════════════
async function compressMemoryNode(state) {
    // Only compress every 2 questions to avoid LLM overhead on every turn
    if (state.questionsAsked % 2 !== 0 || state.chatHistory.length < 4) {
        return {}; // no-op
    }
    console.log("[Interview] Compressing memory…");

    const transcript = buildCompactHistory(state.chatHistory, 6);
    const llm = makeLLM(0.1, 150);
    const response = await llm.invoke([
        new SystemMessage(
            "Summarize this interview transcript in 3 sentences. Capture the topics covered and the candidate's performance level. Be concise — this is internal context, not a report."
        ),
        new HumanMessage(transcript)
    ]);

    const summary = response.content.trim();
    console.log(`[Interview] Memory compressed. Summary: "${summary.substring(0, 80)}…"`);

    // Keep only the last 2 turns (4 messages) in chatHistory — rest is in summary
    const trimmedHistory = state.chatHistory.slice(-4);

    return {
        conversationSummary: summary,
        chatHistory: trimmedHistory,
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 5 — GENERATE QUESTION  (uses streaming + prefetched context)
// ═══════════════════════════════════════════════════════════════════
async function generateQuestionNode(state) {
    // If the candidate went off-topic or was nervous, we repeat the current question.
    // If they skipped, or it's a normal answer, we advance.
    const isRepeat = state.intent === INTENTS.NERVOUS || state.intent === INTENTS.IRRELEVANT;
    const nextQ = isRepeat ? state.questionsAsked : state.questionsAsked + 1;

    console.log(`[Interview] Generating Q#${nextQ} (${state.difficultyLevel}) ${isRepeat ? '[REPEAT]' : ''}`);

    // Use prefetched context if available, otherwise fetch now
    const context = state.prefetchedContext
        || await getContext(state.sessionId, "core concepts definitions key ideas", 3);

    // Use summary + last 2 turns instead of full history → ~80% fewer tokens
    const summaryBlock = state.conversationSummary
        ? `INTERVIEW SUMMARY SO FAR:\n${state.conversationSummary}\n`
        : "";
    const recentHistory = buildCompactHistory(state.chatHistory, 2);
    const isLast = nextQ === state.maxQuestions;
    const socket = getSocket(state.sessionId);

    const systemPrompt =
        `You are a curious, senior engineer interviewing a candidate. Ask question #${nextQ} of ${state.maxQuestions}.

DIFFICULTY: ${state.difficultyLevel}
- easy   → basic recall, definitions, simple facts
- medium → understanding, cause-and-effect, comparisons
- hard   → analysis, tradeoffs, edge cases

SOURCE MATERIAL (base your question on this):
${context}

${summaryBlock}RECENT CONVERSATION (last 2 turns):
${recentHistory}

${isRepeat ?
            `CRITICAL INSTRUCTION: The candidate got sidetracked or was nervous. 
*Gently repeat or slightly rephrase the PREVIOUS question* they were supposed to answer.` :
            `TOPICS ALREADY COVERED (do not repeat): ${state.topicsUsed.slice(-5).join("; ") || "none"}`}

RULES:
- ONE question only. No preamble. No "Question 3:" prefix.
- ${isLast && !isRepeat ? "This is the LAST question. Start with [final_question] then ask naturally." : "Start with [next_question] then ask the question."}
- 1 sentence max, conversational, spoken
- Build naturally on the conversation flow
- No trick or compound questions`;

    let questionText = "";

    // ── Stream to socket if connected ─────────────────────────────
    if (socket) {
        const llm = makeLLM(0.85, 100, true);
        const stream = await llm.stream([
            new SystemMessage(systemPrompt),
            new HumanMessage("Ask the next question.")
        ]);
        for await (const chunk of stream) {
            const token = chunk.content;
            if (token) {
                questionText += token;
                socket.emit("ai_stream", { token, type: "question" });
            }
        }
        socket.emit("ai_stream_end", { type: "question" });
    } else {
        const llm = makeLLM(0.85, 100);
        const response = await llm.invoke([
            new SystemMessage(systemPrompt),
            new HumanMessage("Ask the next question.")
        ]);
        questionText = response.content.trim();
    }

    console.log(`[Interview] Q#${nextQ}: "${questionText.substring(0, 80)}…"`);

    return {
        currentQuestion: questionText,
        questionsAsked: nextQ,
        topicsUsed: !isRepeat ? [...state.topicsUsed, "General concept from: " + context.substring(0, 30)] : state.topicsUsed,
        chatHistory: [...state.chatHistory, new AIMessage(questionText)],
        prefetchedContext: "" // clear it!
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 6 — FINAL REPORT
// ═══════════════════════════════════════════════════════════════════
async function generateFinalReportNode(state) {
    console.log("[Interview] Generating final report…");
    const avg = state.scores.length
        ? (state.scores.reduce((a, b) => a + b, 0) / state.scores.length).toFixed(1)
        : "N/A";

    // Use compressed summary + last 4 turns — not the full history
    const summaryBlock = state.conversationSummary
        ? `INTERVIEW SUMMARY:\n${state.conversationSummary}\n`
        : "";
    const transcript = buildCompactHistory(state.chatHistory, 4);

    const systemPrompt =
        `You are a Senior Technical Recruiter writing an internal interview debrief.

CANDIDATE: ${state.candidateName}
QUESTIONS ASKED: ${state.questionsAsked} of ${state.maxQuestions}
SCORES PER QUESTION: ${state.scores.join(", ")} (out of 10)
AVERAGE SCORE: ${avg}/10
${state.interviewStopped ? "⚠️ NOTE: Interview ended early at the candidate's request." : ""}

GRADE RUBRIC:
A = avg 8.5-10  (exceptional, strong hire)
B = avg 7-8.4   (solid, hire)
C = avg 5-6.9   (borderline, needs more eval)
D = avg 3-4.9   (weak, likely no hire)
F = avg 0-2.9   (very poor, no hire)

${summaryBlock}RECENT TRANSCRIPT:
${transcript}

Write a concise recruiter debrief in markdown. Use EXACTLY these sections:

## Overall Grade
[single letter] — [one sentence justification]

## Key Strengths
- [specific observation]
- [specific observation]

## Areas for Improvement  
- [specific, actionable gap]
- [specific, actionable gap]

## Final Recommendation
[Strong Hire / Hire / Borderline / No Hire] — [one sentence rationale]

## Interviewer Notes
[1-2 sentences of context a recruiter would actually want to know]`;

    const llm = makeLLM(0.3, 350);
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Write the debrief.")
    ]);

    return { finalReport: response.content };
}

// ═══════════════════════════════════════════════════════════════════
// ROUTING
// ═══════════════════════════════════════════════════════════════════

// Decides the very first node to run:
//   - No userAnswer yet → first question → go to generateQuestion
//   - userAnswer present → user replied to a question → go to detectIntent
function routeOnStart(state) {
    return state.userAnswer ? "detectIntent" : "generateQuestion";
}

function afterIntentRoute(state) {
    if (state.intent === INTENTS.NORMAL) {
        // Fan-out: run evaluateAnswer AND prefetchContext in parallel
        return ["evaluateAnswer", "prefetchContext"];
    }
    return "handleEdgeCase";
}

function afterEdgeCaseRoute(state) {
    if (state.interviewStopped) return "generateFinalReport";
    return state.questionsAsked >= state.maxQuestions ? "generateFinalReport" : "generateQuestion";
}

function afterMergeRoute(state) {
    return state.questionsAsked >= state.maxQuestions ? "generateFinalReport" : "generateQuestion";
}

// ═══════════════════════════════════════════════════════════════════
// COMPILE GRAPH
// ═══════════════════════════════════════════════════════════════════
export function createInterviewAgent() {
    const workflow = new StateGraph({ channels: interviewStateChannels })
        .addNode("detectIntent", detectIntentNode)
        .addNode("handleEdgeCase", handleEdgeCaseNode)
        .addNode("evaluateAnswer", evaluateAnswerNode)
        .addNode("prefetchContext", prefetchContextNode)   // ← runs in parallel with evaluateAnswer
        .addNode("adaptDifficulty", adaptDifficultyNode)   // ← join point for parallel branches
        .addNode("compressMemory", compressMemoryNode)
        .addNode("generateQuestion", generateQuestionNode)
        .addNode("generateFinalReport", generateFinalReportNode)

        // ── START: route to generateQuestion (first call) or detectIntent (subsequent) ──
        .addConditionalEdges(START, routeOnStart, {
            generateQuestion: "generateQuestion",
            detectIntent: "detectIntent",
        })
        .addEdge("generateQuestion", END)

        // ── Answer processing: detectIntent fans out to handleEdgeCase OR
        //    to [evaluateAnswer + prefetchContext] simultaneously ──────────
        .addConditionalEdges("detectIntent", afterIntentRoute, {
            evaluateAnswer: "evaluateAnswer",
            prefetchContext: "prefetchContext",
            handleEdgeCase: "handleEdgeCase",
        })

        // ── Edge case path ───────────────────────────────────────────────
        .addConditionalEdges("handleEdgeCase", afterEdgeCaseRoute)

        // ── Parallel branches both feed into adaptDifficulty (join node) ─
        .addEdge("evaluateAnswer", "adaptDifficulty")
        .addEdge("prefetchContext", "adaptDifficulty")

        // ── After join: compress memory then route ───────────────────────
        .addEdge("adaptDifficulty", "compressMemory")
        .addConditionalEdges("compressMemory", afterMergeRoute)

        .addEdge("generateFinalReport", END);

    return workflow.compile({ checkpointer: new MemorySaver() });
}
