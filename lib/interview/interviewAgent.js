import { StateGraph, START, END } from "@langchain/langgraph";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { createLLMWithFallback } from "../llm.js";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { interviewLog, llmLog } from "../logger.js";
import { detectRateLimitError } from "../llm.js";
import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, "..");
dotenv.config();

// ═══════════════════════════════════════════════════════════════════
// LLM LOGGER
// ═══════════════════════════════════════════════════════════════════
const globalLLMLogger = {
    runs: new Map(),
    handleLLMStart(llm, prompts, runId) {
        this.runs.set(runId, Date.now());
        llmLog.info({ provider: 'groq', model: 'llama-3.3-70b-versatile' }, 'LLM request started');
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

        llmLog.info({ durationMs: duration, promptTokens: pTokens, completionTokens: cTokens, totalTokens: tTokens }, 'LLM response received');
    },
    handleLLMError(error, runId) {
        const duration = Date.now() - (this.runs.get(runId) || Date.now());
        this.runs.delete(runId);
        const rateLimitType = detectRateLimitError(error);
        if (rateLimitType) {
            llmLog.error({ durationMs: duration, rateLimitType, err: error?.message }, 'LLM rate limit / quota error');
        } else {
            llmLog.error({ durationMs: duration, err: error?.message || 'Unknown error' }, 'LLM call failed');
        }
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
    "lets_move_on":      "assets/tts/lets_move_on.mp3",
    "great_answer":      "assets/tts/great_answer.mp3",
    "good_effort":       "assets/tts/good_effort.mp3",
    "take_your_time":    "assets/tts/take_your_time.mp3",
    "no_worries":        "assets/tts/no_worries.mp3",
    "interesting":       "assets/tts/interesting.mp3",
    "next_question":     "assets/tts/next_question.mp3",
    "final_question":    "assets/tts/final_question.mp3",
    "thats_okay":        "assets/tts/thats_okay.mp3",
    "thanks_for_time":   "assets/tts/thanks_for_time.mp3",
    "interview_intro":   "assets/tts/interview_intro.mp3",
    "interview_outro":   "assets/tts/interview_outro.mp3",
    "interview_stopped": "assets/tts/interview_stopped.mp3",
    "out_of_context":    "assets/tts/out_of_context.mp3",
};

export function checkTTSCache(phraseKey) {
    const relativePath = TTS_PHRASE_CACHE[phraseKey];
    if (!relativePath) return null;
    const filePath = path.resolve(PROJECT_ROOT, relativePath);
    if (fs.existsSync(filePath)) return filePath;
    return null;
}

export function parseTTSResponse(responseText) {
    const tags = [];
    const tagRegex = /\[(\w+)\]/g;
    let match;
    while ((match = tagRegex.exec(responseText)) !== null) {
        const phraseKey = match[1];
        if (TTS_PHRASE_CACHE[phraseKey]) tags.push(phraseKey);
    }

    const uniquePart = responseText.replace(/\[(\w+)\]/g, "").trim();

    return {
        phraseKeys: tags,
        phraseKey: tags.length > 0 ? tags[0] : null,
        uniquePart: stripMarkdownForTTS(uniquePart),
    };
}

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
// SOCKET REGISTRY
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
    timeGreeting:        { value: (x, y) => y ?? x, default: () => "Hello" },
    prefetchedContext:   { value: (x, y) => y ?? x, default: () => "" },
    repeatCount:         { value: (x, y) => y ?? x, default: () => 0 },
    answerQuality:       { value: (x, y) => y ?? x, default: () => "normal" },

    // ── NEW FIELDS ──────────────────────────────────────────────────
    // Curriculum: pre-planned topic list generated at session start
    questionPlan:        { value: (x, y) => y ?? x, default: () => [] },
    // Topic of the current question (from curriculum plan)
    topicTag:            { value: (x, y) => y ?? x, default: () => "" },
    // Specific concept the candidate missed (drives follow-up generation)
    keyConceptMissed:    { value: (x, y) => y ?? x, default: () => "" },
    // Per-topic score map: { "arrays": [8, 6], "async": [5] }
    topicScores:         { value: (x, y) => y ?? x, default: () => ({}) },
    // Whether a follow-up has been asked for the current planned question
    followUpAsked:       { value: (x, y) => y ?? x, default: () => false },
    // Tells generateQuestionNode to ask a follow-up instead of the next planned question
    followUpFlag:        { value: (x, y) => y ?? x, default: () => false },
    // Consecutive wrong answers — triggers supportive tone in question generation
    struggleStreak:      { value: (x, y) => y ?? x, default: () => 0 },
    // Cross-session user profile context (injected at start if user is authenticated)
    userProfileContext:   { value: (x, y) => y ?? x, default: () => "" },
};

const INTENTS = {
    STOP:       "stop",
    SKIP:       "skip",
    NERVOUS:    "nervous",
    UNWELL:     "unwell",
    IRRELEVANT: "irrelevant",
    NORMAL:     "normal"
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
// NODE 0 — PLAN CURRICULUM
// Runs ONCE at session start. 1 LLM call generates all N topic slots
// so every subsequent generateQuestion call uses the targeted plan.
// ═══════════════════════════════════════════════════════════════════
async function planCurriculumNode(state) {
    interviewLog.info({ sessionId: state.sessionId, maxQuestions: state.maxQuestions }, 'Planning curriculum from document');
    const context = await getContext(state.sessionId, "main topics key concepts overview summary", 5);

    const userProfileBlock = state.userProfileContext
        ? `\nCANDIDATE HISTORY (from previous interviews):\n${state.userProfileContext}\n\nUse this to personalize the curriculum — focus MORE on their weak areas and LESS on topics they've already mastered.\n`
        : "";

    const llm = makeLLM(0.3, 350);
    const response = await llm.invoke([
        new SystemMessage(`You are planning a technical interview curriculum based on a document.

DOCUMENT EXCERPTS:
${context}
${userProfileBlock}
Create exactly ${state.maxQuestions} interview question slots in a logical easy → hard progression.
Return ONLY a valid JSON array, no extra text:

[
  { "topic": "<specific topic from document>", "difficulty": "easy",   "angle": "<what aspect to probe>" },
  { "topic": "<specific topic>",               "difficulty": "medium", "angle": "<what aspect to probe>" },
  ...
]

Rules:
- Topics MUST come from the document content above
- Each topic must be distinct — no repetition
- Difficulty: first 1-2 easy, middle medium, last 1-2 hard
- angle = the specific aspect to test (e.g. "definition", "how it works", "tradeoffs", "real-world use")${userProfileBlock ? "\n- Prioritize the candidate's WEAK AREAS when choosing topics and angles" : ""}`),
        new HumanMessage("Generate the curriculum plan now.")
    ]);

    let plan;
    try {
        const clean = response.content.trim().replace(/```json\n?|```/g, "");
        plan = JSON.parse(clean);
        if (!Array.isArray(plan)) throw new Error("Not an array");
    } catch (err) {
        interviewLog.warn({ err: err.message }, 'Curriculum parse failed, using default plan');
        plan = Array.from({ length: state.maxQuestions }, (_, i) => ({
            topic: "core concepts",
            difficulty: i < 2 ? "easy" : i < state.maxQuestions - 1 ? "medium" : "hard",
            angle: "understanding"
        }));
    }

    // Ensure exactly maxQuestions items
    plan = plan.slice(0, state.maxQuestions);
    while (plan.length < state.maxQuestions) {
        plan.push({ topic: "general knowledge", difficulty: "medium", angle: "understanding" });
    }

    interviewLog.info({ plan: plan.map(p => `${p.topic}(${p.difficulty})`).join(' -> ') }, 'Curriculum ready');
    return { questionPlan: plan };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 1 — PROCESS ANSWER
// Combined intent detection + evaluation in 1 LLM call.
// Now also: silence detection bypass, topicTag, keyConceptMissed,
// followUpFlag decision, struggleStreak, topicScores.
// ═══════════════════════════════════════════════════════════════════
async function processAnswerNode(state) {
    interviewLog.info({ questionNum: state.questionsAsked, answerLength: state.userAnswer?.length }, 'Processing answer');

    // ── SILENCE DETECTION: < 5 real words → treat as nervous, skip LLM ──
    const wordCount = state.userAnswer.trim().split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount < 5) {
        interviewLog.info({ wordCount }, 'Short response detected, routing as nervous');
        return {
            intent: INTENTS.NERVOUS,
            answerQuality: "skipped",
            followUpFlag: false,
            followUpAsked: state.followUpAsked,
            struggleStreak: state.struggleStreak || 0,
            evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: "[take_your_time]", nextDifficulty: "same" },
            chatHistory: [
                ...state.chatHistory,
                new HumanMessage(state.userAnswer),
                new AIMessage("[take_your_time]")
            ],
            scores: state.scores
        };
    }

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
Classify the candidate's intent. Be VERY lenient — any attempt to answer = "normal".

Intent options:
- "normal"     → ANY genuine attempt to answer, even if wrong, incomplete, or confused
- "skip"       → explicitly wants to skip (e.g. "skip", "I don't know", "next question", "pass")
- "stop"       → wants to end interview entirely (e.g. "stop", "end interview", "quit")
- "nervous"    → expressing anxiety/freezing (e.g. "I'm blanking", "I'm nervous", "I can't think")
- "unwell"     → health-related (e.g. "I'm sick", "I need a break", "I don't feel well")
- "irrelevant" → ONLY if clearly ZERO connection to the interview topic

Voice-to-text note: "saw the interview" = "stop", "dont no" = "I don't know" = skip.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK 2 — EVALUATE (only if intent = "normal")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score the answer against the reference material.

Scoring:
- 1-3: Wrong or completely off-base
- 4-6: Partial — shows some understanding but missing key parts
- 7-8: Solid — mostly correct, good understanding
- 9-10: Excellent — complete, demonstrates deep understanding

answerQuality: "correct" (7-10) | "partial" (4-6) | "wrong" (1-3) | "skipped" (non-normal)

feedback:
- ONE sentence, natural spoken language, specific to what they said
- NO bracketed tags — never include [great_answer] etc.
- Warm but honest, like a thoughtful human interviewer

nextDifficulty: "easier" if score < 5, "same" if 5-7, "harder" if >= 8

topicTag: The specific topic this question was about (e.g. "React hooks", "database indexing").
          Use the topic from the question context, be concise (2-4 words max).

keyConceptMissed: If answerQuality is "partial" or "wrong", name the ONE specific concept
                  the candidate missed or got wrong (e.g. "closure scope", "O(n) complexity").
                  Empty string if answer was correct or intent was not normal.

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
  "nextDifficulty": "<easier|same|harder>",
  "topicTag": "<topic label, 2-4 words>",
  "keyConceptMissed": "<specific missed concept or empty string>"
}`;

    const llm = makeLLM(0.1, 300);
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Process this answer now.")
    ]);

    let data;
    try {
        const clean = response.content.trim().replace(/```json\n?|```/g, "");
        data = JSON.parse(clean);
    } catch (err) {
        interviewLog.warn({ err: err.message }, 'ProcessAnswer JSON parse failed, using defaults');
        data = {
            intent: "normal", score: 5, accuracy: 5, clarity: 5, depth: 5,
            answerQuality: "partial", feedback: "Got it, let's continue.",
            nextDifficulty: "same", topicTag: state.topicTag || "general", keyConceptMissed: ""
        };
    }

    const validatedIntent = Object.values(INTENTS).includes(data.intent) ? data.intent : INTENTS.NORMAL;

    // Strip any tags the LLM may have added to feedback
    const cleanFeedback = (data.feedback || "")
        .replace(/\[(great_answer|good_effort|interesting|thats_okay|lets_move_on|take_your_time|no_worries|next_question|final_question|interview_intro|interview_outro|interview_stopped|out_of_context)\]/gi, "")
        .trim();

    // TTS quality tag
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

    // ── FOLLOW-UP DECISION ──────────────────────────────────────────
    // Trigger follow-up only when: normal intent + partial answer + haven't already followed up
    const shouldFollowUp =
        validatedIntent === INTENTS.NORMAL &&
        data.answerQuality === "partial" &&
        !state.followUpAsked;

    // ── STRUGGLE STREAK ─────────────────────────────────────────────
    // Track consecutive wrong answers → generateQuestionNode uses supportive tone
    let newStruggleStreak = state.struggleStreak || 0;
    if (validatedIntent === INTENTS.NORMAL) {
        newStruggleStreak = data.answerQuality === "wrong" ? newStruggleStreak + 1 : 0;
    }

    // ── PER-TOPIC SCORES ─────────────────────────────────────────────
    const tag = data.topicTag || state.topicTag || "general";
    const newTopicScores = { ...state.topicScores };
    if (validatedIntent === INTENTS.NORMAL) {
        newTopicScores[tag] = [...(newTopicScores[tag] || []), data.score || 0];
    }

    const newScores = validatedIntent === INTENTS.NORMAL
        ? [...state.scores, evalData.score]
        : state.scores;

    interviewLog.info(
        { intent: validatedIntent, quality: data.answerQuality, score: data.score, followUp: shouldFollowUp, struggleStreak: newStruggleStreak, topic: tag, keyConceptMissed: data.keyConceptMissed || null },
        'Answer processed'
    );

    // Emit real-time feedback to socket
    const socket = getSocket(state.sessionId);
    if (socket && validatedIntent === INTENTS.NORMAL) {
        socket.emit("ai_feedback", {
            feedback: cleanFeedback,
            score: evalData.score,
            answerQuality: data.answerQuality,
            topicTag: tag,
            followUpcoming: shouldFollowUp
        });
    }

    return {
        intent: validatedIntent,
        answerQuality: data.answerQuality,
        evaluation: evalData,
        topicTag: tag,
        keyConceptMissed: data.keyConceptMissed || "",
        followUpFlag: shouldFollowUp,
        followUpAsked: shouldFollowUp ? true : state.followUpAsked,
        struggleStreak: newStruggleStreak,
        topicScores: newTopicScores,
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
    const CACHE_ONLY_INTENTS = {
        [INTENTS.STOP]:       "interview_stopped",
        [INTENTS.SKIP]:       "thats_okay",
        [INTENTS.NERVOUS]:    "take_your_time",
        [INTENTS.UNWELL]:     "no_worries",
        [INTENTS.IRRELEVANT]: "out_of_context",
    };

    const cachedKey = CACHE_ONLY_INTENTS[state.intent];

    if (cachedKey) {
        const isLastQuestion = state.questionsAsked >= state.maxQuestions;
        const stoppedByIntent = state.intent === INTENTS.STOP || state.intent === INTENTS.UNWELL;
        // UX rule: on the last question, any non-normal intent should close immediately.
        const shouldEndNow = isLastQuestion || stoppedByIntent;
        interviewLog.info({ intent: state.intent, cachedTTS: cachedKey, shouldEndNow }, 'Edge case handled');

        if (stoppedByIntent) {
            const socket = getSocket(state.sessionId);
            const cachedFile = checkTTSCache(cachedKey);
            if (socket && cachedFile) {
                socket.emit("play_tts", {
                    key: cachedKey,
                    audio: fs.readFileSync(cachedFile).toString("base64")
                });
            }
        }

        const feedbackText = shouldEndNow
            ? `[${cachedKey}] [thanks_for_time]`
            : `[${cachedKey}]`;

        return {
            chatHistory: [
                ...state.chatHistory,
                new HumanMessage(state.userAnswer),
                new AIMessage(feedbackText)
            ],
            interviewStopped: shouldEndNow,
            evaluation: {
                score: 0, accuracy: 0, clarity: 0, depth: 0,
                feedback: feedbackText, nextDifficulty: "same"
            }
        };
    }

    // Fallback: unexpected intent → single LLM redirection
    interviewLog.info({ intent: state.intent }, 'Unexpected intent, generating LLM redirection');
    const socket = getSocket(state.sessionId);
    const llm = makeLLM(0.85, 120, !!socket);
    const prompt = `The candidate said: "${state.userAnswer}". Write one brief, professional sentence acknowledging this and redirecting to the interview.`;

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
        evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: responseText, nextDifficulty: "same" }
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 3 — PREFETCH CONTEXT
// Runs in PARALLEL with adaptDifficulty.
// Now uses the curriculum plan for a targeted topic query instead of
// a generic "core concepts" string, reducing irrelevant RAG chunks.
// ═══════════════════════════════════════════════════════════════════
async function prefetchContextNode(state) {
    try {
        // If a follow-up is pending, prefetch for the CURRENT topic (same question slot)
        // Otherwise prefetch for the NEXT planned topic
        const planIdx = state.followUpFlag
            ? Math.max(0, state.questionsAsked - 1)  // current slot (0-indexed)
            : state.questionsAsked;                   // next slot (questionsAsked not yet incremented)

        const planItem = state.questionPlan[planIdx];
        const query = planItem
            ? `${planItem.topic} ${planItem.angle}`
            : "core concepts definitions key ideas";

        interviewLog.debug({ query, planIdx }, 'Prefetching context');
        const ctx = await getContext(state.sessionId, query, 3);
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
        interviewLog.info({ from: state.difficultyLevel, to: newLevel }, 'Difficulty adjusted');
    }

    return { difficultyLevel: newLevel };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 4b — UPDATE SUMMARY (replaces compressMemoryNode)
// Deterministic — no LLM call. Builds conversationSummary from
// already-known scores + topics and trims chatHistory to bound size.
// ═══════════════════════════════════════════════════════════════════
function updateSummaryNode(state) {
    // Build a compact, LLM-readable summary from existing state data
    const summaryLines = state.scores.map((score, i) => {
        const topic = state.topicsUsed[i] || `Q${i + 1}`;
        const indicator = score >= 7 ? "✓" : score >= 4 ? "~" : "✗";
        return `${indicator} ${topic}: ${score}/10`;
    });

    const conversationSummary = summaryLines.length > 0
        ? summaryLines.join(" | ")
        : "";

    // Keep last 6 messages (3 Q&A pairs) to keep context compact
    const trimmedHistory = state.chatHistory.length > 6
        ? state.chatHistory.slice(-6)
        : state.chatHistory;

    return {
        conversationSummary,
        chatHistory: trimmedHistory,
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 5 — GENERATE QUESTION
// Handles two paths:
//   A) followUpFlag=true  → probe deeper on same topic, no questionsAsked increment
//   B) followUpFlag=false → read next slot from questionPlan, increment questionsAsked
// ═══════════════════════════════════════════════════════════════════
async function generateQuestionNode(state) {
    const socket = getSocket(state.sessionId);

    // ── PATH A: FOLLOW-UP ──────────────────────────────────────────
    if (state.followUpFlag) {
        interviewLog.info({ questionNum: state.questionsAsked, topic: state.topicTag }, 'Generating follow-up question');

        const context = state.prefetchedContext
            || await getContext(state.sessionId, state.topicTag || state.currentQuestion, 3);

        const systemPrompt = `You are a senior technical interviewer probing a candidate's partial answer.

ORIGINAL QUESTION:
${state.currentQuestion}

CANDIDATE'S ANSWER:
"${state.userAnswer}"

KEY CONCEPT THEY MISSED:
${state.keyConceptMissed || "they gave an incomplete answer — probe for missing details"}

TOPIC: ${state.topicTag}
SOURCE MATERIAL:
${context}

RULES:
- Ask ONE follow-up question that targets the specific gap
- Stay on the SAME topic — do NOT introduce new concepts
- Guide them toward the missed concept without giving it away
- Be warm, curious, and conversational — not interrogative
- 1-2 sentences max, no tag prefix
- Example good follow-up: "You mentioned X — can you walk me through how that actually works under the hood?"`;

        let questionText = "";

        if (socket) {
            const llm = makeLLM(0.85, 100, true);
            const stream = await llm.stream([
                new SystemMessage(systemPrompt),
                new HumanMessage("Ask the follow-up question.")
            ]);
            for await (const chunk of stream) {
                const token = chunk.content;
                if (token) {
                    questionText += token;
                    socket.emit("ai_stream", { token, type: "follow_up" });
                }
            }
            socket.emit("ai_stream_end", { type: "follow_up" });
        } else {
            const llm = makeLLM(0.85, 100);
            const response = await llm.invoke([
                new SystemMessage(systemPrompt),
                new HumanMessage("Ask the follow-up question.")
            ]);
            questionText = response.content.trim();
        }

        interviewLog.info({ followUp: questionText.substring(0, 80) }, 'Follow-up generated');

        return {
            currentQuestion: questionText,
            followUpFlag: false,
            // questionsAsked NOT incremented — same curriculum slot
            chatHistory: [...state.chatHistory, new AIMessage(questionText)],
            prefetchedContext: ""
        };
    }

    // ── PATH B: PLANNED QUESTION ───────────────────────────────────

    // Handle nervous/irrelevant repeat logic (rephrase same question)
    const isRepeat = state.intent === INTENTS.NERVOUS || state.intent === INTENTS.IRRELEVANT;
    let effectiveRepeat = isRepeat;
    let newRepeatCount = isRepeat ? (state.repeatCount || 0) + 1 : 0;

    if (isRepeat && (state.repeatCount || 0) >= 2) {
        interviewLog.info({ repeatCount: state.repeatCount }, 'Max repeats reached, force-advancing');
        effectiveRepeat = false;
        newRepeatCount = 0;
    }

    const nextQ = effectiveRepeat ? state.questionsAsked : state.questionsAsked + 1;
    const isFirstQuestion = nextQ === 1;
    const isLast = nextQ === state.maxQuestions;

    // Read from curriculum plan (0-indexed: nextQ-1)
    const planIdx = nextQ - 1;
    const planItem = state.questionPlan[planIdx] || {
        topic: "core concepts",
        difficulty: state.difficultyLevel,
        angle: "understanding"
    };

    // Use prefetched context (targeted to this topic) or fetch now
    const context = state.prefetchedContext
        || await getContext(state.sessionId, `${planItem.topic} ${planItem.angle}`, 3);

    const recentHistory = buildCompactHistory(state.chatHistory, 2);
    const summaryBlock = state.conversationSummary
        ? `INTERVIEW PROGRESS:\n${state.conversationSummary}\n`
        : "";

    // Supportive tone if candidate got 2+ wrong answers in a row
    const supportiveTone = (state.struggleStreak || 0) >= 2;

    let tagInstruction = "";
    if (isFirstQuestion) {
        tagInstruction = `Start with [interview_intro] then say "${state.timeGreeting}, ${state.candidateName}! Let's get started." then ask the first question naturally.`;
    } else if (isLast && !effectiveRepeat) {
        tagInstruction = "This is the LAST question. Start with [final_question] then ask.";
    } else if (effectiveRepeat) {
        tagInstruction = "IMPORTANT: Gently rephrase the SAME question — candidate didn't answer it. No tag.";
    } else {
        tagInstruction = "Start with [next_question] then ask the question.";
    }

    interviewLog.info({ questionNum: nextQ, maxQuestions: state.maxQuestions, topic: planItem.topic, difficulty: planItem.difficulty, repeat: effectiveRepeat, supportiveTone }, 'Generating question');

    const userCtx = state.userProfileContext
        ? `\nCANDIDATE BACKGROUND: ${state.userProfileContext}\n`
        : "";

    const systemPrompt = `You are a senior engineer interviewing a candidate. Ask question #${nextQ} of ${state.maxQuestions}.

TOPIC TO COVER: ${planItem.topic}
ANGLE: ${planItem.angle}
DIFFICULTY: ${planItem.difficulty}
- easy   → basic definitions, simple facts
- medium → understanding, cause-and-effect, how it works
- hard   → analysis, tradeoffs, edge cases, real-world design

SOURCE MATERIAL:
${context}
${userCtx}
${summaryBlock}RECENT CONVERSATION:
${recentHistory || "No prior conversation yet."}

${!effectiveRepeat ? `TOPICS ALREADY COVERED: ${state.topicsUsed.slice(-5).join("; ") || "none"}` : ""}
${supportiveTone ? "\nTONE: Candidate is struggling. Be especially encouraging and use simpler, clearer phrasing." : ""}

RULES:
- ${tagInstruction}
- ONE question only, no preamble, no "Question X:" prefix
- 1-2 sentences max, conversational, spoken naturally
- ${effectiveRepeat ? "Rephrase the same question in a different, clearer way" : "Target the topic and angle above — build naturally on the conversation"}
- No compound or trick questions`;

    let questionText = "";

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

    interviewLog.info({ questionNum: nextQ, question: questionText.substring(0, 80) }, 'Question generated');

    return {
        currentQuestion: questionText,
        questionsAsked: nextQ,
        topicTag: planItem.topic,
        repeatCount: newRepeatCount,
        followUpAsked: false,   // Reset follow-up flag for this new planned question
        topicsUsed: !effectiveRepeat
            ? [...state.topicsUsed, planItem.topic]
            : state.topicsUsed,
        chatHistory: [...state.chatHistory, new AIMessage(questionText)],
        prefetchedContext: ""
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 6 — FINAL REPORT
// Pre-aggregates all stats before the LLM call to reduce token usage
// and give the model richer, pre-computed signal.
// ═══════════════════════════════════════════════════════════════════
async function generateFinalReportNode(state) {
    interviewLog.info({ questionsAsked: state.questionsAsked, maxQuestions: state.maxQuestions, stopped: state.interviewStopped }, 'Generating final report');

    const scores = state.scores;

    // ── Pre-aggregate stats (no extra LLM call needed) ──
    const avg = scores.length
        ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1)
        : 0;
    const best  = scores.length ? Math.max(...scores) : 0;
    const worst = scores.length ? Math.min(...scores) : 0;
    const trend = scores.length >= 2
        ? (scores[scores.length - 1] > scores[0] ? "improving"
            : scores[scores.length - 1] < scores[0] ? "declining"
            : "steady")
        : "N/A";
    const grade = parseFloat(avg) >= 8.5 ? "A"
        : parseFloat(avg) >= 7 ? "B"
        : parseFloat(avg) >= 5 ? "C"
        : parseFloat(avg) >= 3 ? "D" : "F";

    // ── Per-topic breakdown from topicScores ──
    const topicBreakdown = Object.entries(state.topicScores || {})
        .map(([topic, topicScoreArr]) => {
            const topicAvg = topicScoreArr.length
                ? (topicScoreArr.reduce((a, b) => a + b, 0) / topicScoreArr.length).toFixed(1)
                : "N/A";
            const icon = parseFloat(topicAvg) >= 7 ? "✅" : parseFloat(topicAvg) >= 4 ? "⚠️" : "❌";
            return `${icon} ${topic}: ${topicAvg}/10`;
        }).join("\n") || "No per-topic data available.";

    const transcript = buildCompactHistory(state.chatHistory, 4);

    const systemPrompt = `You are a Senior Technical Recruiter writing an internal interview debrief.

CANDIDATE: ${state.candidateName}
QUESTIONS ASKED: ${state.questionsAsked} of ${state.maxQuestions}
${state.interviewStopped ? "⚠️ Interview ended early at candidate's request.\n" : ""}
PERFORMANCE STATS:
- Average Score : ${avg}/10
- Best Score    : ${best}/10
- Worst Score   : ${worst}/10
- Trend         : ${trend}
- Implied Grade : ${grade}

PER-TOPIC BREAKDOWN:
${topicBreakdown}

GRADE RUBRIC:
A = 8.5-10  (exceptional, strong hire)
B = 7-8.4   (solid, hire)
C = 5-6.9   (borderline)
D = 3-4.9   (weak, likely no hire)
F = 0-2.9   (very poor, no hire)

RECENT TRANSCRIPT:
${transcript}

Write a concise debrief in markdown:

## Overall Grade
[letter] — [one sentence]

## Key Strengths
- [observation]
- [observation]

## Areas for Improvement
- [specific gap with topic name]
- [specific gap with topic name]

## Final Recommendation
[Strong Hire / Hire / Borderline / No Hire] — [rationale]

## Interviewer Notes
[1-2 sentences of context, mention the performance trend]`;

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
    // Has a user answer → evaluate it
    if (state.userAnswer) return "processAnswer";
    // First invocation — curriculum not yet planned
    if (!state.questionPlan || state.questionPlan.length === 0) return "planCurriculum";
    // Curriculum exists, no answer → generate first question
    return "generateQuestion";
}

function afterProcessAnswerRoute(state) {
    if (state.intent !== INTENTS.NORMAL) return "handleEdgeCase";
    // Normal: run adaptDifficulty and prefetchContext in parallel
    return ["adaptDifficulty", "prefetchContext"];
}

function afterEdgeCaseRoute(state) {
    if (state.interviewStopped) return "generateFinalReport";
    if (state.questionsAsked >= state.maxQuestions) return "generateFinalReport";
    return "generateQuestion";
}

function afterSummaryRoute(state) {
    // followUpFlag=true means questionsAsked hasn't incremented → never go to report early
    if (!state.followUpFlag && state.questionsAsked >= state.maxQuestions) {
        return "generateFinalReport";
    }
    return "generateQuestion";
}

// ═══════════════════════════════════════════════════════════════════
// COMPILE GRAPH
// ═══════════════════════════════════════════════════════════════════
export async function createInterviewAgent(checkpointer) {
    // If no checkpointer provided, create a PostgresSaver from env
    if (!checkpointer && process.env.SUPABASE_DB_URL) {
        checkpointer = PostgresSaver.fromConnString(process.env.SUPABASE_DB_URL);
        await checkpointer.setup();
        interviewLog.info('PostgresSaver checkpointer initialized');
    }

    const workflow = new StateGraph({ channels: interviewStateChannels })
        .addNode("planCurriculum",      planCurriculumNode)
        .addNode("processAnswer",       processAnswerNode)
        .addNode("handleEdgeCase",      handleEdgeCaseNode)
        .addNode("prefetchContext",     prefetchContextNode)
        .addNode("adaptDifficulty",     adaptDifficultyNode)
        .addNode("updateSummary",       updateSummaryNode)
        .addNode("generateQuestion",    generateQuestionNode)
        .addNode("generateFinalReport", generateFinalReportNode)

        // ── Entry ──────────────────────────────────────────────────
        // First invocation: planCurriculum → generateQuestion → END
        // Subsequent:       processAnswer → (branching below)
        .addConditionalEdges(START, routeOnStart, {
            planCurriculum:  "planCurriculum",
            generateQuestion: "generateQuestion",
            processAnswer:   "processAnswer",
        })
        .addEdge("planCurriculum",  "generateQuestion")
        .addEdge("generateQuestion", END)

        // ── After processAnswer ────────────────────────────────────
        // Non-normal intent → handleEdgeCase
        // Normal → parallel: adaptDifficulty + prefetchContext
        .addConditionalEdges("processAnswer", afterProcessAnswerRoute, {
            adaptDifficulty: "adaptDifficulty",
            prefetchContext: "prefetchContext",
            handleEdgeCase:  "handleEdgeCase",
        })

        // ── Edge case ──────────────────────────────────────────────
        .addConditionalEdges("handleEdgeCase", afterEdgeCaseRoute)

        // ── Parallel fan-in → updateSummary (deterministic, no LLM) ──
        .addEdge("adaptDifficulty", "updateSummary")
        .addEdge("prefetchContext", "updateSummary")

        // ── After summary: follow-up or next question or report ───
        .addConditionalEdges("updateSummary", afterSummaryRoute)

        .addEdge("generateFinalReport", END);

    const compileOpts = {};
    if (checkpointer) compileOpts.checkpointer = checkpointer;

    return workflow.compile(compileOpts);
}
