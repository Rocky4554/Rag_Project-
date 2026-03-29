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
const PROJECT_ROOT = path.resolve(__dirname, "../..");
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
    if (!docs || docs.length === 0) {
        interviewLog.warn({ sessionId, query }, 'RAG retrieval returned no documents');
        const fallback = "[NO DOCUMENT CONTENT FOUND — only ask questions based on what you can confirm from available context]";
        ragCache.set(key, fallback);
        return fallback;
    }
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

    // ── ACOUSTIC CONTEXT (Improvement #4) ───────────────────────────
    // Injected from the WebRTC/STT layer before LangGraph is invoked.
    // Allows the LLM to calibrate feedback tone based on how the candidate sounded.
    timeToAnswer:        { value: (x, y) => y ?? x, default: () => 0 },    // ms from AI finishing to STT final
    utteranceDurationMs: { value: (x, y) => y ?? x, default: () => 0 },    // how long candidate spoke
    fillerWordCount:     { value: (x, y) => y ?? x, default: () => 0 },    // count of uh/um/er
    bargedIn:            { value: (x, y) => y ?? x, default: () => false }, // candidate interrupted AI

    // LLM-decided follow-up angle (Improvement #3): specific probe direction for next question
    followUpAngle:       { value: (x, y) => y ?? x, default: () => "" },
};

const INTENTS = {
    // Improvement #1: dedicated answer_attempt separates routing from grading
    ANSWER:     "answer_attempt",    // genuine attempt to answer → gradeAnswerNode
    STOP:       "stop",
    SKIP:       "skip",
    NERVOUS:    "nervous",
    UNWELL:     "unwell",
    IRRELEVANT: "irrelevant",
    CONFUSED:   "confused",
    META:       "meta",
    // Improvement #1: new semantic intents for human-like routing
    THINKING:   "thinking_out_loud", // "hmm let me think..." → backchannel, don't evaluate
    CUTOFF:     "premature_cutoff",  // STT cut mid-sentence → keep listening
    NORMAL:     "normal",            // backward-compat alias for answer_attempt
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
- Topics MUST come ONLY from the document content above — NEVER use your general knowledge to invent topics
- If the document context is empty or says "[NO DOCUMENT CONTENT FOUND]", generate topics using ONLY generic labels like "document overview", "key details", "main concepts" — do NOT hallucinate specific technical topics like React, JavaScript, etc.
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
            topic: "document content",
            difficulty: i < 2 ? "easy" : i < state.maxQuestions - 1 ? "medium" : "hard",
            angle: "ask about what is in the uploaded document only"
        }));
    }

    // Ensure exactly maxQuestions items
    plan = plan.slice(0, state.maxQuestions);
    while (plan.length < state.maxQuestions) {
        plan.push({ topic: "document content", difficulty: "medium", angle: "ask about what is in the uploaded document only" });
    }

    interviewLog.info({ plan: plan.map(p => `${p.topic}(${p.difficulty})`).join(' -> ') }, 'Curriculum ready');
    return { questionPlan: plan };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 1a — ROUTER  (Improvement #1)
// Fast, dedicated intent classifier — runs BEFORE any grading.
// Uses a minimal token budget so it returns in ~150ms.
// If intent = answer_attempt → gradeAnswerNode
// If intent = thinking_out_loud / premature_cutoff → backchannelNode
// Everything else → handleEdgeCaseNode
// ═══════════════════════════════════════════════════════════════════
async function routerNode(state) {
    interviewLog.info({ questionNum: state.questionsAsked, answerLength: state.userAnswer?.length }, 'Router classifying intent');

    // ── FAST-PATH: 1-2 word responses handled by regex (no LLM) ───────
    const wordCount = state.userAnswer.trim().split(/\s+/).filter(w => w.length > 0).length;
    if (wordCount <= 2) {
        const lower = state.userAnswer.trim().toLowerCase().replace(/[^a-z\s]/g, "").trim();
        if (/^(stop|end|quit|finish|done|goodbye|bye|exit)$/.test(lower)) {
            interviewLog.info({ wordCount, lower }, 'Fast-path STOP detected');
            return { intent: INTENTS.STOP };
        }
        if (/^(skip|pass|next)$/.test(lower)) {
            interviewLog.info({ wordCount, lower }, 'Fast-path SKIP detected');
            return { intent: INTENTS.SKIP };
        }
        // Single filler word = thinking out loud ("um", "uh", "hmm")
        if (/^(um|uh|er|hmm|hm|yeah|okay|ok)$/.test(lower)) {
            interviewLog.info({ wordCount, lower }, 'Fast-path THINKING detected');
            return { intent: INTENTS.THINKING };
        }
        interviewLog.info({ wordCount }, 'Fast-path NERVOUS detected (very short, non-stop)');
        return { intent: INTENTS.NERVOUS };
    }

    // ── Fast-path: 3-6 word "I don't know" / skip signals ────────────────
    // Handles common STT variants: "dont know", "no idea", "not sure", etc.
    // Prevents the LLM from misclassifying these as STOP at low token budgets.
    if (wordCount <= 6) {
        const lowerNorm = state.userAnswer.trim().toLowerCase().replace(/[^a-z\s]/g, "").trim();
        if (/\b(dont know|no idea|have no idea|not sure|cant answer|i dont know|i have no idea)\b/.test(lowerNorm)) {
            interviewLog.info({ wordCount, lowerNorm }, 'Fast-path SKIP detected (dont-know phrase)');
            return { intent: INTENTS.SKIP };
        }
    }

    // ── Improvement #2: detect filler-heavy short responses as "thinking" ─
    // e.g. "um let me think" or "so uh" — 3-4 words with trailing filler
    if (wordCount <= 4 && state.fillerWordCount > 0) {
        const lower = state.userAnswer.trim().toLowerCase();
        if (/\b(um|uh|er|so|well|hmm)\s*[.!?]?\s*$/.test(lower)) {
            interviewLog.info({ wordCount, fillerWordCount: state.fillerWordCount }, 'Short filler response → THINKING');
            return { intent: INTENTS.THINKING };
        }
    }

    // ── LLM router: intent classification only, minimal token budget ────
    const systemPrompt = `You are classifying a candidate's speech in a technical interview.
Respond with ONLY the intent word — nothing else, no punctuation.

CURRENT QUESTION: "${state.currentQuestion}"
CANDIDATE SAID: "${state.userAnswer}"

Intent options (pick exactly one):
- answer_attempt   → ANY genuine attempt to answer, even wrong, partial, or rambling
- stop             → wants to END the whole interview ("end the interview", "I want to stop", "let's quit")
- skip             → wants to skip THIS question ("I don't know", "pass", "next question", "move on")
- nervous          → anxiety or mental freeze ("I'm blanking", "I'm so nervous", "I can't think")
- unwell           → health issue ("I'm sick", "I need a break", "I don't feel well")
- confused         → doesn't understand the QUESTION itself ("what do you mean?", "can you rephrase?", "I have no idea what you're asking")
- meta             → asking about interview process ("how many questions left?", "what's my score?", "how am I doing?")
- irrelevant       → completely off-topic, chatting, jokes, or unrelated questions
- thinking_out_loud → still forming thoughts, not ready to answer ("hmm let me think...", "well so I'm thinking...", "okay so um...")
- premature_cutoff  → sentence clearly cut off mid-thought with no conclusion ("so when you have a list and you want to", "the main thing is that")

Voice-to-text corrections:
- "and the interview" / "saw the interview" → stop
- "dont no" / "no idea" → skip
- "can you repeat" / "say that again" → confused`;

    const llm = makeLLM(0.0, 20); // Temperature 0, tiny budget — single intent word
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Classify now.")
    ]);

    const raw = response.content.trim().toLowerCase().replace(/[^a-z_]/g, "");
    const validIntents = Object.values(INTENTS);
    const intent = validIntents.includes(raw) ? raw : INTENTS.ANSWER;

    interviewLog.info({ intent, raw, wordCount, answer: state.userAnswer.substring(0, 80) }, 'Router intent classified');
    return { intent };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 1b — GRADE ANSWER  (Improvement #1, #3, #4)
// Only runs when routerNode classified intent as answer_attempt/normal.
// Does NOT re-detect intent — evaluation only.
// Adds: acoustic context injection (#4), LLM-driven follow-up (#3).
// ═══════════════════════════════════════════════════════════════════
async function gradeAnswerNode(state) {
    interviewLog.info({ questionNum: state.questionsAsked, answerLength: state.userAnswer?.length }, 'Grading answer');

    const context = await getContext(state.sessionId, state.currentQuestion);

    // ── Improvement #4: Build acoustic behavioral context for the LLM ──
    // Injected from worker.js via sessionBridge so the model can adjust
    // feedback tone based on HOW the candidate answered, not just what they said.
    const acousticParts = [];
    if (state.timeToAnswer > 4000) {
        acousticParts.push(`Paused ${Math.round(state.timeToAnswer / 1000)}s before answering — was thinking it through`);
    }
    if (state.utteranceDurationMs > 0 && state.utteranceDurationMs < 5000) {
        acousticParts.push(`Gave a very brief answer (${Math.round(state.utteranceDurationMs / 1000)}s)`);
    } else if (state.utteranceDurationMs > 35000) {
        acousticParts.push(`Gave a detailed response (${Math.round(state.utteranceDurationMs / 1000)}s)`);
    }
    if (state.fillerWordCount >= 3) {
        acousticParts.push(`Used ${state.fillerWordCount} filler words (um/uh) — likely uncertain or hesitant`);
    }
    if (state.bargedIn) {
        acousticParts.push("Interrupted the AI — was eager and confident");
    }
    const acousticContext = acousticParts.length > 0
        ? `\nBEHAVIORAL CONTEXT (from real-time audio):\n${acousticParts.join(". ")}.\nUse this to calibrate feedback tone — be extra encouraging for hesitant answers; acknowledge confidence for barge-ins.\n`
        : "";

    // ── Improvement #3: LLM decides follow-up + angle ──────────────────
    const followUpInstruction = state.followUpAsked
        ? "requires_follow_up: false (already asked a follow-up for this question slot)\nfollow_up_angle: \"\""
        : `requires_follow_up: true if a follow-up would significantly deepen understanding:
  - score 4-8 AND answer was substantive but missed a key aspect → follow up to probe the gap
  - score >= 8 AND answer showed real mastery → follow up to probe edge cases or depth
  - score < 4 → false (answer was fundamentally wrong; not worth probing a wrong foundation)
follow_up_angle: Specific probe direction (what exactly to ask next). Empty string if requires_follow_up is false.
  Example: "Probe whether they understand the difference between shallow and deep copy"`;

    const systemPrompt = `You are a senior engineer evaluating a technical interview answer.
Intent has already been classified as "answer_attempt" — the candidate IS trying to answer.
Your job is ONLY to score and evaluate. Do not re-classify intent.

INTERVIEW QUESTION:
${state.currentQuestion}

CANDIDATE'S ANSWER (voice-transcribed):
"${state.userAnswer}"

REFERENCE MATERIAL (from the uploaded document):
${context}
${acousticContext}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 1-3: Wrong or completely off-base
- 4-6: Partial — shows some understanding but missing key parts
- 7-8: Solid — mostly correct, good understanding
- 9-10: Excellent — complete, demonstrates deep understanding

answerQuality: "correct" (7-10) | "partial" (4-6) | "wrong" (1-3)

feedback:
- ONE sentence, natural spoken language, specific to what they said
- NO bracketed tags — never write [great_answer] etc. in feedback
- If behavioral context says they were hesitant/struggling, be warm and encouraging
- If they barged in enthusiastically, acknowledge their eagerness

nextDifficulty: "easier" if score < 5, "same" if 5-7, "harder" if >= 8

topicTag: The specific topic this question tested (2-4 words, e.g. "React hooks", "database indexing").

keyConceptMissed: The ONE specific concept missed if answerQuality is "partial" or "wrong".
                  Empty string if answer was correct.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FOLLOW-UP DECISION (Improvement #3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${followUpInstruction}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — return ONLY valid JSON, no extra text:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "score": <1-10>,
  "accuracy": <1-10>,
  "clarity": <1-10>,
  "depth": <1-10>,
  "answerQuality": "<correct|partial|wrong>",
  "feedback": "<one natural spoken sentence>",
  "nextDifficulty": "<easier|same|harder>",
  "topicTag": "<topic label, 2-4 words>",
  "keyConceptMissed": "<missed concept or empty string>",
  "requires_follow_up": <true|false>,
  "follow_up_angle": "<specific probe direction or empty string>"
}`;

    const llm = makeLLM(0.1, 320);
    const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage("Grade this answer now.")
    ]);

    let data;
    try {
        const clean = response.content.trim().replace(/```json\n?|```/g, "");
        data = JSON.parse(clean);
    } catch (err) {
        interviewLog.warn({ err: err.message }, 'GradeAnswer JSON parse failed, using defaults');
        data = {
            score: 5, accuracy: 5, clarity: 5, depth: 5,
            answerQuality: "partial", feedback: "Got it, let's continue.",
            nextDifficulty: "same", topicTag: state.topicTag || "general",
            keyConceptMissed: "", requires_follow_up: false, follow_up_angle: ""
        };
    }

    // Strip any bracket tags the LLM may have accidentally added to feedback
    const cleanFeedback = (data.feedback || "")
        .replace(/\[(great_answer|good_effort|interesting|thats_okay|lets_move_on|take_your_time|no_worries|next_question|final_question|interview_intro|interview_outro|interview_stopped|out_of_context)\]/gi, "")
        .trim();

    // Prepend a TTS quality tag so worker can play a cached phrase before the feedback
    let qualityTag = "";
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

    const taggedFeedback = cleanFeedback ? `${qualityTag} ${cleanFeedback}` : "";

    const evalData = {
        score: data.score || 0,
        accuracy: data.accuracy || 0,
        clarity: data.clarity || 0,
        depth: data.depth || 0,
        feedback: taggedFeedback,
        nextDifficulty: data.nextDifficulty || "same"
    };

    // ── Improvement #3: LLM-driven follow-up ───────────────────────────
    const shouldFollowUp = (data.requires_follow_up === true) && !state.followUpAsked;
    const followUpAngle = data.follow_up_angle || "";

    // ── STRUGGLE STREAK ─────────────────────────────────────────────
    const newStruggleStreak = data.answerQuality === "wrong"
        ? (state.struggleStreak || 0) + 1
        : 0;

    // ── PER-TOPIC SCORES ─────────────────────────────────────────────
    const tag = data.topicTag || state.topicTag || "general";
    const newTopicScores = { ...state.topicScores };
    newTopicScores[tag] = [...(newTopicScores[tag] || []), data.score || 0];

    interviewLog.info({
        quality: data.answerQuality, score: data.score, nextDifficulty: evalData.nextDifficulty,
        followUp: shouldFollowUp, followUpAngle: followUpAngle.substring(0, 60),
        struggleStreak: newStruggleStreak, topic: tag,
        timeToAnswer: state.timeToAnswer, fillerWordCount: state.fillerWordCount,
    }, 'Answer graded');

    // Emit real-time feedback to socket
    const socket = getSocket(state.sessionId);
    if (socket) {
        socket.emit("ai_feedback", {
            feedback: cleanFeedback,
            score: evalData.score,
            answerQuality: data.answerQuality,
            topicTag: tag,
            followUpcoming: shouldFollowUp
        });
    }

    return {
        intent: INTENTS.ANSWER,
        answerQuality: data.answerQuality,
        evaluation: evalData,
        topicTag: tag,
        keyConceptMissed: data.keyConceptMissed || "",
        followUpFlag: shouldFollowUp,
        followUpAngle: followUpAngle,
        followUpAsked: shouldFollowUp ? true : state.followUpAsked,
        struggleStreak: newStruggleStreak,
        topicScores: newTopicScores,
        chatHistory: [
            ...state.chatHistory,
            new HumanMessage(state.userAnswer),
            new AIMessage(taggedFeedback || state.userAnswer)
        ],
        scores: [...state.scores, evalData.score]
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 1c — BACKCHANNEL  (Improvement #1)
// Runs when routerNode classified intent as thinking_out_loud or premature_cutoff.
// Does NOT advance question state. Returns to END immediately so the worker
// can speak a brief acknowledgment and go back to listening.
// ═══════════════════════════════════════════════════════════════════
function backchannelNode(state) {
    if (state.intent === INTENTS.CUTOFF) {
        // STT cut the candidate off mid-sentence — don't speak, just wait.
        interviewLog.info({ sessionId: state.sessionId }, 'Premature cutoff — keeping state, waiting for more speech');
        return {
            evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: "", nextDifficulty: "same" }
        };
    }

    // thinking_out_loud — acknowledge and wait without advancing the question
    interviewLog.info({ sessionId: state.sessionId }, 'Thinking out loud — sending backchannel phrase');
    return {
        evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: "[take_your_time]", nextDifficulty: "same" }
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE 2 — HANDLE EDGE CASES
// ═══════════════════════════════════════════════════════════════════
async function handleEdgeCaseNode(state) {
    // ── CONFUSED: rephrase the current question more simply ─────────
    if (state.intent === INTENTS.CONFUSED) {
        interviewLog.info({ sessionId: state.sessionId }, 'Candidate confused — rephrasing question');
        const llm = makeLLM(0.5, 160);
        const response = await llm.invoke([
            new SystemMessage("You are a friendly interviewer. The candidate didn't understand the question. Rephrase it more simply and clearly in 1-2 sentences. Start with a brief acknowledgment like 'No problem!' or 'Of course!'. Keep the same topic and intent but use simpler language."),
            new HumanMessage(`Original question: "${state.currentQuestion}"\n\nRephrase more simply now.`)
        ]);
        const rephrased = response.content.trim();
        return {
            currentQuestion: rephrased,
            chatHistory: [...state.chatHistory, new HumanMessage(state.userAnswer), new AIMessage(rephrased)],
            interviewStopped: false,
            evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: rephrased, nextDifficulty: "same" }
        };
    }

    // ── META: answer questions about interview progress ─────────────
    if (state.intent === INTENTS.META) {
        const remaining = Math.max(0, state.maxQuestions - state.questionsAsked);
        const avgScore = state.scores.length
            ? (state.scores.reduce((a, b) => a + b, 0) / state.scores.length).toFixed(1)
            : null;
        const lower = state.userAnswer.toLowerCase();

        let answer;
        if (lower.match(/\b(remain|left|more|how many|questions)\b/)) {
            answer = remaining === 0
                ? "This is actually the last question — you're almost done!"
                : `You have ${remaining} question${remaining !== 1 ? 's' : ''} remaining after this one.`;
        } else if (lower.match(/\b(score|doing|performance|result|how am)\b/)) {
            answer = avgScore
                ? `Your average score so far is ${avgScore} out of 10. Keep it up!`
                : "We've just started, so no scores yet. Let's continue!";
        } else if (lower.match(/\b(topic|about|this question|what are we)\b/)) {
            answer = state.topicTag
                ? `We're currently discussing ${state.topicTag}. Let's continue!`
                : "We're covering topics from your document. Let's continue!";
        } else {
            answer = avgScore
                ? `You have ${remaining} question${remaining !== 1 ? 's' : ''} remaining and your average score is ${avgScore}/10. Let's keep going!`
                : `You have ${remaining} question${remaining !== 1 ? 's' : ''} remaining. Let's keep going!`;
        }

        interviewLog.info({ sessionId: state.sessionId, remaining, avgScore, answer }, 'Meta question answered');
        return {
            chatHistory: [...state.chatHistory, new HumanMessage(state.userAnswer), new AIMessage(answer)],
            interviewStopped: false,
            evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: answer, nextDifficulty: "same" }
        };
    }

    // ── IRRELEVANT: acknowledge briefly and redirect back ───────────
    if (state.intent === INTENTS.IRRELEVANT) {
        interviewLog.info({ sessionId: state.sessionId }, 'Off-topic comment — generating redirect');
        const socket = getSocket(state.sessionId);
        const llm = makeLLM(0.7, 120, !!socket);
        const prompt = `The candidate said something off-topic during an interview: "${state.userAnswer}". Write 1-2 brief, friendly sentences that: acknowledge their comment naturally, then redirect them back to the interview. Be warm, not dismissive. End naturally with something like "Now, let's get back to our interview."`;

        let redirect = "";
        if (socket) {
            const stream = await llm.stream([
                new SystemMessage("Write natural spoken responses. No markdown, no lists."),
                new HumanMessage(prompt)
            ]);
            for await (const chunk of stream) {
                const token = chunk.content;
                if (token) {
                    redirect += token;
                    socket.emit("ai_stream", { token, type: "redirect" });
                }
            }
            socket.emit("ai_stream_end", { type: "redirect" });
        } else {
            const res = await llm.invoke([
                new SystemMessage("Write natural spoken responses. No markdown, no lists."),
                new HumanMessage(prompt)
            ]);
            redirect = res.content.trim();
        }

        return {
            chatHistory: [...state.chatHistory, new HumanMessage(state.userAnswer), new AIMessage(redirect)],
            interviewStopped: false,
            evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: redirect, nextDifficulty: "same" }
        };
    }

    const CACHE_ONLY_INTENTS = {
        [INTENTS.STOP]:    "interview_stopped",
        [INTENTS.SKIP]:    "thats_okay",
        [INTENTS.NERVOUS]: "take_your_time",
        [INTENTS.UNWELL]:  "no_worries",
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

        // Improvement #3: use the LLM-suggested probe angle if available
        const probeAngle = state.followUpAngle ||
            (state.keyConceptMissed
                ? `probe the specific concept they missed: "${state.keyConceptMissed}"`
                : "probe for deeper understanding or edge cases");

        const systemPrompt = `You are a senior technical interviewer probing a candidate's answer.

ORIGINAL QUESTION:
${state.currentQuestion}

CANDIDATE'S ANSWER:
"${state.userAnswer}"

FOLLOW-UP ANGLE (what to probe):
${probeAngle}

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

        interviewLog.info({ followUp: questionText.substring(0, 150), topic: state.topicTag, difficulty: state.difficultyLevel }, 'Follow-up generated');

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
        // No greeting here — speakIntro() in the agent worker already handles the
        // personalised greeting. Just ask the first question directly.
        tagInstruction = "Ask the first question directly. Do not add any greeting or preamble.";
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

    interviewLog.info({ questionNum: nextQ, question: questionText.substring(0, 150), topic: planItem.topic, difficulty: state.difficultyLevel }, 'Question generated');

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
    // Has a user answer → classify intent first (routerNode)
    if (state.userAnswer) return "routerNode";
    // First invocation — curriculum not yet planned
    if (!state.questionPlan || state.questionPlan.length === 0) return "planCurriculum";
    // Curriculum exists, no answer → generate first question
    return "generateQuestion";
}

// Improvement #1: after fast intent classification, route to the right node
function afterRouterRoute(state) {
    // answer_attempt / normal → full evaluation in gradeAnswerNode
    if (state.intent === INTENTS.ANSWER || state.intent === INTENTS.NORMAL) return "gradeAnswer";
    // thinking_out_loud / premature_cutoff → brief backchannel, keep same question
    if (state.intent === INTENTS.THINKING || state.intent === INTENTS.CUTOFF) return "backchannel";
    // Everything else (stop/skip/nervous/unwell/confused/meta/irrelevant) → edge case handler
    return "handleEdgeCase";
}

// After gradeAnswerNode: always run adaptDifficulty + prefetchContext in parallel
function afterGradeRoute() {
    return ["adaptDifficulty", "prefetchContext"];
}

function afterEdgeCaseRoute(state) {
    if (state.interviewStopped) return "generateFinalReport";
    if (state.questionsAsked >= state.maxQuestions) return "generateFinalReport";
    // confused/meta/irrelevant are handled entirely inside handleEdgeCaseNode —
    // response already built, no new question needed. Next user speech restarts from processAnswer.
    if (state.intent === INTENTS.CONFUSED || state.intent === INTENTS.META || state.intent === INTENTS.IRRELEVANT) return END;
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
        // Improvement #1: split into fast router + dedicated grader + backchannel
        .addNode("routerNode",          routerNode)
        .addNode("gradeAnswer",         gradeAnswerNode)
        .addNode("backchannel",         backchannelNode)
        .addNode("handleEdgeCase",      handleEdgeCaseNode)
        .addNode("prefetchContext",     prefetchContextNode)
        .addNode("adaptDifficulty",     adaptDifficultyNode)
        .addNode("updateSummary",       updateSummaryNode)
        .addNode("generateQuestion",    generateQuestionNode)
        .addNode("generateFinalReport", generateFinalReportNode)

        // ── Entry ──────────────────────────────────────────────────
        // First invocation: planCurriculum → generateQuestion → END
        // Subsequent:       routerNode → (branching below)
        .addConditionalEdges(START, routeOnStart, {
            planCurriculum:  "planCurriculum",
            generateQuestion: "generateQuestion",
            routerNode:      "routerNode",
        })
        .addEdge("planCurriculum",  "generateQuestion")
        .addEdge("generateQuestion", END)

        // ── After router: grade | backchannel | edge case ──────────
        .addConditionalEdges("routerNode", afterRouterRoute, {
            gradeAnswer:   "gradeAnswer",
            backchannel:   "backchannel",
            handleEdgeCase: "handleEdgeCase",
        })

        // ── Backchannel → END immediately (worker speaks phrase, stays on same Q) ──
        .addEdge("backchannel", END)

        // ── After grading: always run adaptDifficulty + prefetchContext in parallel ──
        .addConditionalEdges("gradeAnswer", afterGradeRoute, {
            adaptDifficulty: "adaptDifficulty",
            prefetchContext: "prefetchContext",
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
