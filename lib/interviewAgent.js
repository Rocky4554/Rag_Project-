import { StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
import { createLLMWithFallback } from "./llm.js";
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
    // Extract ALL tags from the response
    const tags = [];
    const tagRegex = /\[(\w+)\]/g;
    let match;
    while ((match = tagRegex.exec(responseText)) !== null) {
        const phraseKey = match[1];
        if (TTS_PHRASE_CACHE[phraseKey]) {
            tags.push(phraseKey);
        }
    }

    // Remove all tags to get the unique part
    const uniquePart = responseText
        .replace(/\[(\w+)\]/g, "")
        .trim();

    return {
        phraseKeys: tags,  // Array of all valid tags found
        phraseKey: tags.length > 0 ? tags[0] : null,  // First tag (backward compatibility)
        uniquePart: stripMarkdownForTTS(uniquePart),
    };
}

// Force-strip any tag from evaluation feedback (safety check)
export function stripEvaluationTags(feedback) {
    return feedback.replace(/\[(great_answer|good_effort|interesting)\]/gi, "").trim();
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
    sessionId:           { value: (x, y) => y ?? x, default: () => "" },
    chatHistory:         { value: (x, y) => y ?? x, default: () => [] },
    conversationSummary: { value: (x, y) => y ?? x, default: () => "" },
    currentQuestion:     { value: (x, y) => y ?? x, default: () => "" },
    userAnswer:          { value: (x, y) => y ?? x, default: () => "" },
    evaluation:          { value: (x, y) => y ?? x, default: () => null },
    difficultyLevel:     { value: (x, y) => y ?? x, default: () => "medium" },
    questionsAsked:      { value: (x, y) => y ?? x, default: () => 0 },
    maxQuestions:        { value: (x, y) => y ?? x, default: () => 5 },
    topicsUsed:          { value: (x, y) => y ?? x, default: () => [] },
    finalReport:         { value: (x, y) => y ?? x, default: () => null },
    interviewStopped:    { value: (x, y) => y ?? x, default: () => false },
    intent:              { value: (x, y) => y ?? x, default: () => "normal" },
    scores:              { value: (x, y) => y ?? x, default: () => [] },
    candidateName:       { value: (x, y) => y ?? x, default: () => "there" },
    prefetchedContext:   { value: (x, y) => y ?? x, default: () => "" },
    // NEW: tracks repeated questions to prevent infinite loops (max 2 repeats)
    repeatCount:         { value: (x, y) => y ?? x, default: () => 0 },
    // NEW: "correct" | "partial" | "wrong" | "skipped"
    answerQuality:       { value: (x, y) => y ?? x, default: () => "normal" },
};

const INTENTS = {
    STOP:      "stop",
    SKIP:      "skip",
    NERVOUS:   "nervous",
    UNWELL:    "unwell",
    IRRELEVANT:"irrelevant",
    NORMAL:    "normal"
};

// ═══════════════════════════════════════════════════════════════════
// LLM FACTORY
// ═══════════════════════════════════════════════════════════════════
function makeLLM(temperature = 0.7, maxTokens = 200, streaming = false) {
    return createLLMWithFallback({
        provider: "groq",
        temperature,
        maxTokens,
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
// NODE 1 — PROCESS ANSWER (combined intent detection + evaluation)
// Replaces the old detectIntentNode + evaluateAnswerNode (2 LLM calls → 1)
// ═══════════════════════════════════════════════════════════════════
async function processAnswerNode(state) {
    console.log(`[Interview] Processing answer for Q#${state.questionsAsked}…`);
    const context = await getContext(state.sessionId, state.currentQuestion);

    const systemPrompt = `You are a senior engineer conducting a technical interview.

INTERVIEW QUESTION:
${state.currentQuestion}

CANDIDATE'S ANSWER (voice-transcribed, may have typos or speech recognition errors):
"${state.userAnswer}"

REFERENCE MATERIAL (from the uploaded document):
${context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 1 — DETECT INTENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classify the candidate's intent. Be VERY lenient — any attempt to answer the question (even wrong, confused, or partially related) = "normal".

Intent options:
- "normal"     → ANY genuine attempt to answer, even if wrong, incomplete, or confused
- "skip"       → explicitly wants to skip or pass (e.g. "skip", "I don't know", "next question", "pass the question")
- "stop"       → wants to end the interview entirely (e.g. "stop", "end interview", "quit", "I want to stop")
- "nervous"    → expressing anxiety/stress/freezing (e.g. "I'm blanking", "I'm nervous", "I can't think")
- "unwell"     → health-related, needs break (e.g. "I'm sick", "I need a break", "I don't feel well")
- "irrelevant" → ONLY if clearly talking about something with ZERO connection to the interview
                 Examples of truly irrelevant: "what is the weather today", "tell me a joke",
                 "what did you eat for lunch", "can you speak Hindi", "my dog is sick"
                 NOT irrelevant: wrong answers, confused answers, partially related answers

Voice-to-text note: "saw the interview" = "stop", "dont no" = "don't know" = skip, "past the question" = "pass the question" = skip.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 2 — EVALUATE (only if intent = "normal")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score the answer against the reference material.

Scoring guide:
- 1-3: Wrong, completely off-base, or just guessing
- 4-6: Partially correct, shows some understanding but missing key parts
- 7-8: Solid, mostly correct, good understanding
- 9-10: Excellent, complete, demonstrates deep understanding

answerQuality:
- "correct"  → score 7-10
- "partial"  → score 4-6
- "wrong"    → score 1-3
- "skipped"  → use for non-normal intents

feedback rules:
- Write as if SPEAKING to the candidate (natural, warm, conversational)
- ONE sentence, specific to what they actually said
- NO bracketed tags like [great_answer] — never add them
- Sound like a thoughtful human interviewer
- Good: "You're on the right track with X, but the key part is actually Y."
- Good: "That's the right idea — the PAN number specifically helps with tax identification."
- Bad: "Great answer!" / "Good effort!" / "[great_answer] Nice job."

nextDifficulty: "easier" if score < 5, "same" if 5-7, "harder" if >= 8

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — return ONLY valid JSON, no extra text:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "intent": "<normal|skip|stop|nervous|unwell|irrelevant>",
  "score": <1-10 or 0 for non-normal>,
  "accuracy": <1-10 or 0>,
  "clarity": <1-10 or 0>,
  "depth": <1-10 or 0>,
  "answerQuality": "<correct|partial|wrong|skipped>",
  "feedback": "<natural spoken sentence, empty string for non-normal>",
  "nextDifficulty": "<easier|same|harder>"
}`;

    const llm = makeLLM(0.1, 250);
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Process this answer now.")
    ]);

    let data;
    try {
        const clean = response.content.trim().replace(/```json\n?|```/g, "");
        data = JSON.parse(clean);
    } catch (err) {
        console.log(`[ProcessAnswer] JSON parse failed: ${err.message}. Defaulting to normal/partial.`);
        data = {
            intent: "normal",
            score: 5,
            accuracy: 5,
            clarity: 5,
            depth: 5,
            answerQuality: "partial",
            feedback: "Got it, let's continue.",
            nextDifficulty: "same"
        };
    }

    // Validate intent
    const validatedIntent = Object.values(INTENTS).includes(data.intent) ? data.intent : INTENTS.NORMAL;

    // Clean up feedback — strip any spurious tags the LLM may have added
    const cleanFeedback = (data.feedback || "")
        .replace(/\[(great_answer|good_effort|interesting|thats_okay|lets_move_on|take_your_time|no_worries|next_question|final_question|interview_intro|interview_outro|interview_stopped|out_of_context)\]/gi, "")
        .trim();

    // Determine answer quality tag for TTS
    let qualityTag = "";
    if (validatedIntent === INTENTS.NORMAL) {
        if (data.answerQuality === "correct" || data.score >= 7) {
            qualityTag = "[great_answer]";
            data.answerQuality = "correct";
        } else if (data.answerQuality === "partial" || data.score >= 4) {
            qualityTag = "[good_effort]";
            data.answerQuality = "partial";
        } else {
            qualityTag = "[interesting]";
            data.answerQuality = "wrong";
        }
    } else {
        data.answerQuality = "skipped";
    }

    // Prepend quality tag to feedback so server.js TTS pipeline plays cached audio first
    const taggedFeedback = validatedIntent === INTENTS.NORMAL && cleanFeedback
        ? `${qualityTag} ${cleanFeedback}`
        : cleanFeedback;

    const evalData = {
        score: data.score || 0,
        accuracy: data.accuracy || 0,
        clarity: data.clarity || 0,
        depth: data.depth || 0,
        feedback: taggedFeedback,
        nextDifficulty: data.nextDifficulty || "same"
    };

    console.log(`[ProcessAnswer] Intent: ${validatedIntent} | Quality: ${data.answerQuality} | Score: ${data.score} | "${state.userAnswer.substring(0, 50)}…"`);

    // Emit answer quality feedback to socket if connected (for real-time UI updates)
    const socket = getSocket(state.sessionId);
    if (socket && validatedIntent === INTENTS.NORMAL) {
        socket.emit("ai_feedback", {
            feedback: cleanFeedback,
            score: evalData.score,
            answerQuality: data.answerQuality
        });
    }

    const newScores = validatedIntent === INTENTS.NORMAL
        ? [...state.scores, evalData.score]
        : state.scores;

    return {
        intent: validatedIntent,
        answerQuality: data.answerQuality,
        evaluation: evalData,
        chatHistory: [
            ...state.chatHistory,
            new HumanMessage(state.userAnswer),
            new AIMessage(taggedFeedback || state.userAnswer)
        ],
        scores: newScores
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 2 — HANDLE EDGE CASES
// ═══════════════════════════════════════════════════════════════════
async function handleEdgeCaseNode(state) {
    // ─────────────────────────────────────────────────────────────────
    // CACHE-ONLY INTENTS: Use pre-recorded audio, no extra LLM call
    // ─────────────────────────────────────────────────────────────────
    const CACHE_ONLY_INTENTS = {
        [INTENTS.STOP]:      "interview_stopped",  // "No problem, let's end here..."
        [INTENTS.SKIP]:      "thats_okay",         // "That's okay, let's move on..."
        [INTENTS.NERVOUS]:   "take_your_time",     // "Take your time, no rush..."
        [INTENTS.UNWELL]:    "no_worries",         // "No worries, we can stop..."
        [INTENTS.IRRELEVANT]:"out_of_context",     // "Let's get back to the question..."
    };

    const cachedKey = CACHE_ONLY_INTENTS[state.intent];

    if (cachedKey) {
        console.log(`[EdgeCase] Intent "${state.intent}" → cached TTS only: ${cachedKey}.mp3`);
        const stopped = state.intent === INTENTS.STOP || state.intent === INTENTS.UNWELL;

        // ─────────────────────────────────────────────────────────────
        // For STOP/UNWELL: emit socket event immediately so the
        // frontend plays the audio BEFORE the report is generated
        // ─────────────────────────────────────────────────────────────
        if (stopped) {
            const socket = getSocket(state.sessionId);
            const cachedFile = checkTTSCache(cachedKey);
            if (socket && cachedFile) {
                console.log(`[EdgeCase] Emitting play_tts via socket for "${cachedKey}" (stop intent)`);
                socket.emit("play_tts", {
                    key: cachedKey,
                    audio: fs.readFileSync(cachedFile).toString("base64")
                });
            }
        }

        // ─────────────────────────────────────────────────────────────
        // SPECIAL CASE: If this is the LAST question and user skips,
        // add a goodbye transition before the report
        // ─────────────────────────────────────────────────────────────
        const isLastQuestion = state.questionsAsked >= state.maxQuestions;
        const needsGoodbye = isLastQuestion && (state.intent === INTENTS.SKIP);

        const feedbackText = needsGoodbye
            ? `[${cachedKey}] [thanks_for_time]`  // Play cache + goodbye
            : `[${cachedKey}]`;

        return {
            chatHistory: [
                ...state.chatHistory,
                new HumanMessage(state.userAnswer),
                new AIMessage(feedbackText)
            ],
            interviewStopped: stopped,
            evaluation: {
                score: 0,
                accuracy: 0,
                clarity: 0,
                depth: 0,
                feedback: feedbackText,
                nextDifficulty: "same"
            }
        };
    }

    // ─────────────────────────────────────────────────────────────────
    // FALLBACK: Unexpected intent → generate with LLM
    // ─────────────────────────────────────────────────────────────────
    console.log(`[EdgeCase] Unexpected intent "${state.intent}", generating LLM response...`);
    const socket = getSocket(state.sessionId);
    const llm = makeLLM(0.85, 120, !!socket);

    const prompt = `The candidate said: "${state.userAnswer}". 
Write one brief, professional sentence acknowledging this and redirecting to the interview.`;

    let responseText = "";

    if (socket) {
        const stream = await llm.stream([
            new SystemMessage("Write natural spoken responses. No markdown, no lists."),
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
            new SystemMessage("Write natural spoken responses. No markdown, no lists."),
            new HumanMessage(prompt)
        ]);
        responseText = response.content.trim();
    }

    return {
        chatHistory: [
            ...state.chatHistory,
            new HumanMessage(state.userAnswer),
            new AIMessage(responseText)
        ],
        interviewStopped: false,
        evaluation: {
            score: 0,
            accuracy: 0,
            clarity: 0,
            depth: 0,
            feedback: responseText,
            nextDifficulty: "same"
        }
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 3 — PREFETCH CONTEXT (runs in PARALLEL with processAnswer)
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

    const newLevel = levels[idx];
    if (newLevel !== state.difficultyLevel) {
        console.log(`[Interview] Difficulty: ${state.difficultyLevel} → ${newLevel}`);
    }

    return { difficultyLevel: newLevel };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 4b — COMPRESS MEMORY (runs after every 2nd answered question)
// ═══════════════════════════════════════════════════════════════════
async function compressMemoryNode(state) {
    if (state.questionsAsked % 2 !== 0 || state.chatHistory.length < 4) {
        return {};
    }

    console.log("[Interview] Compressing memory…");
    const transcript = buildCompactHistory(state.chatHistory, 6);
    const llm = makeLLM(0.1, 150);

    const response = await llm.invoke([
        new SystemMessage(
            "Summarize this interview transcript in 3 sentences. Capture topics covered and candidate performance. Be concise."
        ),
        new HumanMessage(transcript)
    ]);

    const summary = response.content.trim();
    console.log(`[Interview] Memory compressed: "${summary.substring(0, 80)}…"`);

    return {
        conversationSummary: summary,
        chatHistory: state.chatHistory.slice(-4),
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 5 — GENERATE QUESTION
// ═══════════════════════════════════════════════════════════════════
async function generateQuestionNode(state) {
    // ─────────────────────────────────────────────────────────────────
    // Determine if this is a repeat due to irrelevant/nervous intent
    // ─────────────────────────────────────────────────────────────────
    const isRepeat = state.intent === INTENTS.NERVOUS || state.intent === INTENTS.IRRELEVANT;

    // ─────────────────────────────────────────────────────────────────
    // REPEAT LIMIT: If we've already repeated this question twice,
    // force-advance to the next question regardless of intent
    // ─────────────────────────────────────────────────────────────────
    let effectiveRepeat = isRepeat;
    let newRepeatCount = isRepeat ? (state.repeatCount || 0) + 1 : 0;

    if (isRepeat && (state.repeatCount || 0) >= 2) {
        console.log(`[Interview] Max repeats (${state.repeatCount}) reached — force-advancing to next question.`);
        effectiveRepeat = false;
        newRepeatCount = 0;
    }

    const nextQ = effectiveRepeat ? state.questionsAsked : state.questionsAsked + 1;
    const isFirstQuestion = nextQ === 1;
    const isLast = nextQ === state.maxQuestions;

    console.log(`[Interview] Generating Q#${nextQ} (${state.difficultyLevel}) ${effectiveRepeat ? '[REPEAT]' : ''} ${isFirstQuestion ? '[FIRST]' : ''} repeatCount=${newRepeatCount}`);

    // Use prefetched context if available
    const context = state.prefetchedContext
        || await getContext(state.sessionId, "core concepts definitions key ideas", 3);

    const summaryBlock = state.conversationSummary
        ? `INTERVIEW SUMMARY SO FAR:\n${state.conversationSummary}\n`
        : "";
    const recentHistory = buildCompactHistory(state.chatHistory, 2);
    const socket = getSocket(state.sessionId);

    // ─────────────────────────────────────────────────────────────────
    // DYNAMIC TAG SELECTION:
    // - First question: [interview_intro]
    // - Last question:  [final_question]
    // - Repeat:         no tag (just rephrase gently)
    // - Normal:         [next_question]
    // ─────────────────────────────────────────────────────────────────
    let tagInstruction = "";
    if (isFirstQuestion) {
        tagInstruction = "Start with [interview_intro] then ask the first question naturally.";
    } else if (isLast && !effectiveRepeat) {
        tagInstruction = "This is the LAST question. Start with [final_question] then ask.";
    } else if (effectiveRepeat) {
        tagInstruction = "IMPORTANT: Gently rephrase the SAME question — candidate didn't answer it. No tag.";
    } else {
        tagInstruction = "Start with [next_question] then ask the question.";
    }

    const systemPrompt = `You are a senior engineer interviewing a candidate. Ask question #${nextQ} of ${state.maxQuestions}.

DIFFICULTY: ${state.difficultyLevel}
- easy   → basic definitions, simple facts
- medium → understanding, cause-and-effect
- hard   → analysis, tradeoffs, edge cases

SOURCE MATERIAL:
${context}

${summaryBlock}RECENT CONVERSATION:
${recentHistory || "No prior conversation yet."}

${!effectiveRepeat ? `TOPICS ALREADY COVERED: ${state.topicsUsed.slice(-5).join("; ") || "none"}` : ""}

RULES:
- ${tagInstruction}
- ONE question only, no preamble, no "Question X:" prefix
- 1-2 sentences max, conversational, spoken naturally
- ${effectiveRepeat ? "Rephrase the same question in a different, clearer way" : "Build on the conversation flow"}
- No compound or trick questions`;

    let questionText = "";

    // ─────────────────────────────────────────────────────────────────
    // Stream to socket if connected
    // ─────────────────────────────────────────────────────────────────
    if (socket) {
        const llm = makeLLM(0.85, 100, true);
        const stream = await llm.stream([
            new SystemMessage(systemPrompt),
            new HumanMessage("Ask the question.")
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
            new HumanMessage("Ask the question.")
        ]);
        questionText = response.content.trim();
    }

    console.log(`[Interview] Q#${nextQ}: "${questionText.substring(0, 80)}…"`);

    return {
        currentQuestion: questionText,
        questionsAsked: nextQ,
        repeatCount: newRepeatCount,
        topicsUsed: !effectiveRepeat
            ? [...state.topicsUsed, "Concept: " + context.substring(0, 30)]
            : state.topicsUsed,
        chatHistory: [...state.chatHistory, new AIMessage(questionText)],
        prefetchedContext: ""
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

    const summaryBlock = state.conversationSummary
        ? `INTERVIEW SUMMARY:\n${state.conversationSummary}\n`
        : "";
    const transcript = buildCompactHistory(state.chatHistory, 4);

    const systemPrompt = `You are a Senior Technical Recruiter writing an internal interview debrief.

CANDIDATE: ${state.candidateName}
QUESTIONS ASKED: ${state.questionsAsked} of ${state.maxQuestions}
SCORES: ${state.scores.join(", ")} (out of 10)
AVERAGE: ${avg}/10
${state.interviewStopped ? "⚠️ Interview ended early at candidate's request." : ""}

GRADE RUBRIC:
A = 8.5-10  (exceptional, strong hire)
B = 7-8.4   (solid, hire)
C = 5-6.9   (borderline)
D = 3-4.9   (weak, likely no hire)
F = 0-2.9   (very poor, no hire)

${summaryBlock}RECENT TRANSCRIPT:
${transcript}

Write a concise debrief in markdown with these sections:

## Overall Grade
[letter] — [one sentence]

## Key Strengths
- [observation]
- [observation]

## Areas for Improvement
- [specific gap]
- [specific gap]

## Final Recommendation
[Strong Hire / Hire / Borderline / No Hire] — [rationale]

## Interviewer Notes
[1-2 sentences of context]`;

    const llm = makeLLM(0.3, 400);
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Write the debrief.")
    ]);

    return { finalReport: response.content };
}

// ═══════════════════════════════════════════════════════════════════
// ROUTING
// ═══════════════════════════════════════════════════════════════════

function routeOnStart(state) {
    return state.userAnswer ? "processAnswer" : "generateQuestion";
}

function afterProcessAnswerRoute(state) {
    // Edge cases (non-normal intents) go to handleEdgeCase
    if (state.intent !== INTENTS.NORMAL) {
        return "handleEdgeCase";
    }
    // Normal path: adapt difficulty + prefetch context in parallel
    return ["adaptDifficulty", "prefetchContext"];
}

function afterEdgeCaseRoute(state) {
    // If interview was stopped (user said "stop" or "unwell"), generate report immediately
    if (state.interviewStopped) {
        return "generateFinalReport";
    }

    // If user skipped or went off-topic on the LAST question → go to report
    if (state.questionsAsked >= state.maxQuestions) {
        return "generateFinalReport";
    }

    // Otherwise, continue with next/repeat question
    return "generateQuestion";
}

function afterMergeRoute(state) {
    return state.questionsAsked >= state.maxQuestions ? "generateFinalReport" : "generateQuestion";
}

// ═══════════════════════════════════════════════════════════════════
// COMPILE GRAPH
// ═══════════════════════════════════════════════════════════════════
export function createInterviewAgent() {
    const workflow = new StateGraph({ channels: interviewStateChannels })
        .addNode("processAnswer",       processAnswerNode)
        .addNode("handleEdgeCase",      handleEdgeCaseNode)
        .addNode("prefetchContext",     prefetchContextNode)
        .addNode("adaptDifficulty",     adaptDifficultyNode)
        .addNode("compressMemory",      compressMemoryNode)
        .addNode("generateQuestion",    generateQuestionNode)
        .addNode("generateFinalReport", generateFinalReportNode)

        // Start: if there's a user answer to process → processAnswer, else generate first question
        .addConditionalEdges(START, routeOnStart, {
            generateQuestion: "generateQuestion",
            processAnswer:    "processAnswer",
        })
        .addEdge("generateQuestion", END)

        // processAnswer routes to handleEdgeCase or parallel (adaptDifficulty + prefetchContext)
        .addConditionalEdges("processAnswer", afterProcessAnswerRoute, {
            adaptDifficulty: "adaptDifficulty",
            prefetchContext: "prefetchContext",
            handleEdgeCase:  "handleEdgeCase",
        })

        // Edge case → finalReport or generateQuestion
        .addConditionalEdges("handleEdgeCase", afterEdgeCaseRoute)

        // Parallel evaluation fanin: both adaptDifficulty and prefetchContext go to compressMemory
        .addEdge("adaptDifficulty", "compressMemory")
        .addEdge("prefetchContext", "compressMemory")

        // After memory compression → either finalReport or generateQuestion
        .addConditionalEdges("compressMemory", afterMergeRoute)

        .addEdge("generateFinalReport", END);

    return workflow.compile({ checkpointer: new MemorySaver() });
}