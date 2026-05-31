/**
 * deepInterviewAgent.js — Intelligent Orchestrator Interview Agent.
 *
 * A deep-agent redesign of the interview flow. Instead of a fixed regex router
 * feeding a 240-line edge-case God Node, an LLM ORCHESTRATOR reasons over the
 * full interview context every turn and DISPATCHES to a specialized sub-agent:
 *
 *   orchestrator → { evaluate | converse | control | backchannel }
 *
 * It reuses the proven grading / question-generation / curriculum / RAG nodes
 * from interviewAgent.js unchanged, and replaces only the "decide + handle a
 * non-answer" part with intelligent, human-like behaviour.
 *
 * Drop-in compatible: maps decisions back to legacy `intent` values so worker.js
 * and sessionBridge.js need no changes. Enable with INTERVIEW_AGENT_MODE=deep.
 *
 * See dev_docs/deep_interview_agent.md for the full design.
 */

import { StateGraph, START, END } from "@langchain/langgraph";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

import { interviewLog } from "../logger.js";
import {
    interviewStateChannels,
    getContext,
    getSocket,
    makeLLM,
    buildCompactHistory,
    INTENTS,
    planCurriculumNode,
    gradeAnswerNode,
    backchannelNode,
    prefetchContextNode,
    adaptDifficultyNode,
    updateSummaryNode,
    generateQuestionNode,
    generateFinalReportNode,
} from "./interviewAgent.js";

// ═══════════════════════════════════════════════════════════════════
// STATE — existing 32 channels + 8 deep-agent channels
// ═══════════════════════════════════════════════════════════════════
const deepInterviewStateChannels = {
    ...interviewStateChannels,

    dispatch:           { value: (x, y) => y ?? x, default: () => "" },
    subStrategy:        { value: (x, y) => y ?? x, default: () => "" },
    directive:          { value: (x, y) => y ?? x, default: () => "" },
    followWithQuestion: { value: (x, y) => y ?? x, default: () => false },
    needsQuestion:      { value: (x, y) => y ?? x, default: () => false },
    interviewMood:      { value: (x, y) => y ?? x, default: () => "neutral" },
    turnCount:          { value: (x, y) => y ?? x, default: () => 0 },

    // Append reducer — keeps the last 6 non-answer strategies for pattern reading
    conversationPattern: {
        value: (x, y) => (y ? [...(x || []), ...y].slice(-6) : x),
        default: () => [],
    },
};

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════
// Fast/cheap Gemini model for formatting work (meta answers, hints, skips).
// Override the exact API model id with GEMINI_MINI_MODEL if needed.
const GEMINI_MINI_MODEL = process.env.GEMINI_MINI_MODEL || "gemini-3.1-flash-lite";

function makeGeminiMini(temperature = 0.5, maxTokens = 90) {
    return new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: GEMINI_MINI_MODEL,
        temperature,
        maxOutputTokens: maxTokens,
    });
}

// subStrategy → legacy intent (keeps worker.js compatible)
const SUBSTRATEGY_INTENT = {
    encourage_nervous:  INTENTS.NERVOUS,
    redirect_focus:     INTENTS.IRRELEVANT,
    answer_meta:        INTENTS.META,
    give_hint:          INTENTS.HINT,
    acknowledge_break:  INTENTS.BREAK,
    warm_greeting:      INTENTS.GREETING,
    thank:              INTENTS.GRATITUDE,
    rephrase_confused:  INTENTS.CONFUSED,
    skip_question:      INTENTS.SKIP,
};

/** Shared return shape for converse responses. Resets the directive. */
function convReturn(state, feedbackText, needsQuestion, intent, extras = {}) {
    return {
        intent,
        needsQuestion: !!needsQuestion,
        chatHistory: [
            ...state.chatHistory,
            new HumanMessage(state.userAnswer),
            new AIMessage(feedbackText),
        ],
        evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: feedbackText, nextDifficulty: "same" },
        directive: "",
        ...extras,
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE — ORCHESTRATOR (the brain)
// ═══════════════════════════════════════════════════════════════════
async function orchestratorNode(state) {
    const ans = (state.userAnswer || "").trim();
    const lower = ans.toLowerCase().replace(/[^a-z\s]/g, "").trim();
    const wc = ans.split(/\s+/).filter(Boolean).length;
    const turnCount = (state.turnCount || 0) + 1;

    // ── Fast-path: pure filler / thinking (no LLM) ──────────────────
    if (wc <= 2 && /^(um|uh|er|hmm|hm|well|so|like|okay|ok|yeah)$/.test(lower)) {
        interviewLog.info({ sessionId: state.sessionId, dispatch: 'backchannel', via: 'fastpath' }, 'Orchestrator dispatch: backchannel');
        return { dispatch: "backchannel", subStrategy: "thinking", intent: INTENTS.THINKING, turnCount };
    }
    // ── Fast-path: explicit stop (no LLM) ───────────────────────────
    if (wc <= 4 && /\b(stop|quit|end the interview|i want to stop|thats enough|im done|end it)\b/.test(lower)) {
        interviewLog.info({ sessionId: state.sessionId, dispatch: 'control', via: 'fastpath' }, 'Orchestrator dispatch: control');
        return { dispatch: "control", subStrategy: "stop", intent: INTENTS.STOP, turnCount };
    }

    // ── LLM orchestration ───────────────────────────────────────────
    const avg = state.scores.length ? (state.scores.reduce((a, b) => a + b, 0) / state.scores.length).toFixed(1) : "none";
    const pattern = (state.conversationPattern || []).join(" → ") || "none";
    const recent = buildCompactHistory(state.chatHistory, 2);
    const acoustic = [];
    if (state.timeToAnswer > 4000) acoustic.push(`paused ${Math.round(state.timeToAnswer / 1000)}s`);
    if (state.utteranceDurationMs && state.utteranceDurationMs < 3000) acoustic.push("very brief");
    if (state.fillerWordCount >= 3) acoustic.push(`${state.fillerWordCount} fillers`);
    if (state.bargedIn) acoustic.push("interrupted the AI");

    const sys = `You are the brain of an AI interviewer. After every candidate utterance you decide what the interviewer does next — like a real human interviewer reading the room.

SITUATION
- Current question: "${state.currentQuestion}"
- Candidate just said: "${ans}"
- Progress: Q${state.questionsAsked}/${state.maxQuestions} | avg score ${avg}/10 | struggle streak ${state.struggleStreak || 0}
- Recent non-answer pattern: ${pattern}
- Hints given this question: ${state.hintsGiven || 0} | off-topic count: ${state.offTopicCount || 0}
- Acoustics: ${acoustic.join(", ") || "normal"}
- Recent conversation:
${recent || "(interview just started)"}

DECIDE dispatch:
- "evaluate"    → a GENUINE attempt to answer the question (even partial, wrong, or rambling). Default for any real attempt.
- "converse"    → NOT an answer, but a human moment: nervous/blanking, off-topic chit-chat, a meta question about the interview (how many left / my score / how long), asking for a hint, asking to pause, greeting, thanks, confused about the QUESTION, or wanting to skip.
- "control"     → wants to END the interview, or is unwell and cannot continue.
- "backchannel" → still thinking / filler / clearly cut off mid-sentence — tiny ack and wait.

If dispatch="converse", choose subStrategy:
  encourage_nervous | redirect_focus | answer_meta | give_hint | acknowledge_break | warm_greeting | thank | rephrase_confused | skip_question
If dispatch="control", subStrategy: stop | unwell

followWithQuestion: true if AFTER handling we should move to a NEW question (e.g. skip → yes). false if we should respond and WAIT for them to attempt the SAME question (nervous, confused, meta, hint, break, greeting, thanks, redirect → false).

directive: ONE sentence telling the interviewer HOW to respond — acknowledge what they SPECIFICALLY said, calibrate tone to the pattern, sound human, never robotic.

mood: neutral | nervous | frustrated | engaged | lost | confident

Return ONLY JSON:
{"dispatch":"...","subStrategy":"...","followWithQuestion":false,"directive":"...","mood":"..."}`;

    const llm = makeLLM(0, 160);
    const call = llm.invoke([new SystemMessage(sys), new HumanMessage("Decide now. JSON only.")])
        .then(r => JSON.parse(String(r.content).trim().replace(/```json\n?|```/g, "")))
        .catch(() => null);
    const timeout = new Promise(res => setTimeout(() => res(null), 1800));
    const d = await Promise.race([call, timeout]);

    // ── Heuristic fallback ──────────────────────────────────────────
    if (!d || !d.dispatch) {
        const dispatch = wc >= 5 ? "evaluate" : "converse";
        interviewLog.warn({ sessionId: state.sessionId, dispatch, wc }, 'Orchestrator fallback (heuristic)');
        return {
            dispatch,
            subStrategy: dispatch === "converse" ? "redirect_focus" : "",
            intent: dispatch === "evaluate" ? INTENTS.ANSWER : INTENTS.IRRELEVANT,
            directive: "",
            followWithQuestion: false,
            interviewMood: state.interviewMood || "neutral",
            turnCount,
            ...(dispatch === "converse" ? { conversationPattern: ["redirect_focus"] } : {}),
        };
    }

    const dispatch = d.dispatch;
    let intent;
    if (dispatch === "evaluate") intent = INTENTS.ANSWER;
    else if (dispatch === "control") intent = d.subStrategy === "unwell" ? INTENTS.UNWELL : INTENTS.STOP;
    else if (dispatch === "backchannel") intent = d.subStrategy === "cutoff" ? INTENTS.CUTOFF : INTENTS.THINKING;
    else intent = SUBSTRATEGY_INTENT[d.subStrategy] || INTENTS.IRRELEVANT;

    interviewLog.info(
        { sessionId: state.sessionId, dispatch, subStrategy: d.subStrategy, mood: d.mood, directive: (d.directive || "").substring(0, 80) },
        `Orchestrator dispatch: ${dispatch}`
    );

    const patternEntry = dispatch === "converse" ? [d.subStrategy || "converse"]
        : dispatch === "control" ? ["control"] : undefined;

    return {
        dispatch,
        subStrategy: d.subStrategy || "",
        directive: d.directive || "",
        followWithQuestion: !!d.followWithQuestion,
        intent,
        interviewMood: d.mood || state.interviewMood || "neutral",
        lastInteractionType: dispatch,
        turnCount,
        ...(patternEntry ? { conversationPattern: patternEntry } : {}),
    };
}

// ═══════════════════════════════════════════════════════════════════
// NODE — CONVERSE (intelligent human-moment handler)
// ═══════════════════════════════════════════════════════════════════
async function converseNode(state) {
    const strat = state.subStrategy;
    const directive = state.directive || "";
    const intent = SUBSTRATEGY_INTENT[strat] || INTENTS.IRRELEVANT;
    interviewLog.info({ sessionId: state.sessionId, strat }, 'Converse handling turn');

    // ── Tier 1 — no LLM ─────────────────────────────────────────────
    if (strat === "acknowledge_break") {
        return convReturn(state, "Of course — take all the time you need. Just start talking whenever you're ready and we'll pick up right here.", false, intent);
    }
    if (strat === "thank") {
        return convReturn(state, "You're very welcome! Let's keep the momentum going.", false, intent);
    }

    // ── Tier 2 — fast Gemini ────────────────────────────────────────
    if (strat === "answer_meta") {
        const total = state.maxQuestions;
        const answered = state.scores.length;                       // graded answers so far
        const remaining = Math.max(0, total - answered);
        const avg = answered ? (state.scores.reduce((a, b) => a + b, 0) / answered).toFixed(1) : null;
        const res = await makeGeminiMini(0.5, 100).invoke([
            new SystemMessage(`You are an interviewer answering a question about the interview's progress. ${directive}
State these facts accurately and naturally: there are ${total} questions in total, the candidate has answered ${answered} so far, so ${remaining} remain${avg ? `, and their average score is ${avg}/10` : ""}. 1-2 sentences, warm, spoken aloud. No markdown.`),
            new HumanMessage(`They asked: "${state.userAnswer}"`),
        ]).catch(() => ({ content: avg
            ? `There are ${total} questions in total — you've answered ${answered}, so ${remaining} to go, and you're averaging ${avg} out of 10. Doing great, let's keep it up!`
            : `There are ${total} questions in total and you've answered ${answered} so far, so ${remaining} to go. Let's keep going!` }));
        return convReturn(state, String(res.content).trim(), false, intent);
    }

    if (strat === "give_hint") {
        const hints = state.hintsGiven || 0;
        if (hints >= 2) {
            return convReturn(state, "I've already nudged you twice on this one — give it your best shot, or say 'skip' to move on.", false, intent, { hintsGiven: hints });
        }
        const ctx = await getContext(state.sessionId, state.currentQuestion, 3);
        const res = await makeGeminiMini(0.5, 90).invoke([
            new SystemMessage(`You are a supportive interviewer giving hint ${hints + 1} of 2. ${directive} Do NOT reveal the answer — give a conceptual nudge. 1-2 sentences, warm.
REFERENCE MATERIAL: ${ctx?.substring(0, 500) || "none"}`),
            new HumanMessage(`Question: "${state.currentQuestion}". They asked: "${state.userAnswer}"`),
        ]).catch(() => ({ content: "Think about the core principle behind this — what's the fundamental idea at play here?" }));
        return convReturn(state, String(res.content).trim(), false, intent, { hintsGiven: hints + 1 });
    }

    if (strat === "skip_question") {
        const res = await makeGeminiMini(0.5, 70).invoke([
            new SystemMessage(`The candidate is skipping this question. ${directive} One warm sentence — no judgment, signal you're moving on. Under 14 words. No markdown.`),
            new HumanMessage(`Question: "${state.currentQuestion}". They said: "${state.userAnswer}"`),
        ]).catch(() => ({ content: "No worries — let's move on to the next one." }));
        return convReturn(state, String(res.content).trim(), true, intent);
    }

    // ── Tier 3 — rich Groq ──────────────────────────────────────────
    if (strat === "encourage_nervous") {
        const past = state.scores.length >= 1
            ? `They've answered ${state.scores.length} question(s); scores ${state.scores.join(", ")}/10; best ${Math.max(...state.scores)}/10.`
            : "This is early in the interview.";
        const res = await makeLLM(0.7, 90).invoke([
            new SystemMessage(`You are an empathetic human interviewer. The candidate is nervous or blanking. ${directive}
${past}
Ground them in ONE warm sentence. If they have strong past scores, reference that specifically. Don't just say "take your time". Sound like a real person.`),
            new HumanMessage(`They said: "${state.userAnswer}". Current question: "${state.currentQuestion}"`),
        ]).catch(() => ({ content: "Hey, no pressure at all — just talk me through whatever comes to mind." }));
        return convReturn(state, String(res.content).trim(), !!state.followWithQuestion, intent);
    }

    if (strat === "rephrase_confused") {
        const res = await makeLLM(0.5, 120).invoke([
            new SystemMessage(`You are a friendly interviewer. The candidate didn't understand the question. ${directive}
Rephrase it more simply and concretely. Start with a brief acknowledgment like "No problem!". Keep the SAME topic. 1-2 sentences.`),
            new HumanMessage(`Original question: "${state.currentQuestion}". They said: "${state.userAnswer}"`),
        ]).catch(() => ({ content: `No problem! Let me put it another way: ${state.currentQuestion}` }));
        const rephrased = String(res.content).trim();
        // Update currentQuestion so the candidate now answers the rephrased version
        return { ...convReturn(state, rephrased, false, intent), currentQuestion: rephrased };
    }

    if (strat === "warm_greeting") {
        const res = await makeLLM(0.6, 70).invoke([
            new SystemMessage(`You are an interviewer mid-interview. The candidate greeted you. ${directive} Brief warm hello, then immediately steer back to the current question. 1-2 sentences.`),
            new HumanMessage(`They said: "${state.userAnswer}". Pending question: "${state.currentQuestion}"`),
        ]).catch(() => ({ content: "Hey, good to have you! So — back to the question." }));
        return convReturn(state, String(res.content).trim(), !!state.followWithQuestion, intent);
    }

    // ── redirect_focus (off-topic) — default ────────────────────────
    const off = (state.offTopicCount || 0) + 1;
    const lower = state.userAnswer.toLowerCase();

    if (/\b(your name|who are you|are you (a |an )?(robot|ai|human|real|machine)|whats your name)\b/.test(lower)) {
        return convReturn(state, "Ha — I'm your AI interviewer! No fancy name, but I've got great taste in questions. Anyway, back to where we were.", false, intent, { offTopicCount: off });
    }
    if (off >= 3) {
        return convReturn(state, "I genuinely appreciate the chat — but let's make the most of your interview time and refocus on the question.", false, intent, { offTopicCount: off });
    }

    const socket = getSocket(state.sessionId);
    const sysR = `You are a warm, human interviewer. The candidate went off-topic. ${directive}
RULES: 1-2 sentences. Acknowledge what they SPECIFICALLY said, then bridge back naturally ("that actually connects to…", "speaking of which…"). NEVER say "let's stay on topic" or "that's off-topic". Be a real person.`;
    const userR = `They said: "${state.userAnswer}". Current question: "${state.currentQuestion}"`;
    let text = "";
    if (socket) {
        const llm = makeLLM(0.7, 110, true);
        const stream = await llm.stream([new SystemMessage(sysR), new HumanMessage(userR)]);
        for await (const c of stream) {
            if (c.content) { text += c.content; socket.emit("ai_stream", { token: c.content, type: "converse" }); }
        }
        socket.emit("ai_stream_end", { type: "converse" });
    } else {
        const res = await makeLLM(0.7, 110).invoke([new SystemMessage(sysR), new HumanMessage(userR)])
            .catch(() => ({ content: "That's interesting! Let's bring it back to the question, though." }));
        text = String(res.content).trim();
    }
    return convReturn(state, text, false, intent, { offTopicCount: off });
}

// ═══════════════════════════════════════════════════════════════════
// NODE — CONTROL FLOW (stop / unwell → end)
// ═══════════════════════════════════════════════════════════════════
function controlFlowNode(state) {
    const isUnwell = state.subStrategy === "unwell";
    const feedback = isUnwell ? "[no_worries] [thanks_for_time]" : "[interview_stopped] [thanks_for_time]";
    interviewLog.info({ sessionId: state.sessionId, isUnwell }, 'ControlFlow ending interview');
    return {
        intent: isUnwell ? INTENTS.UNWELL : INTENTS.STOP,
        chatHistory: [...state.chatHistory, new HumanMessage(state.userAnswer), new AIMessage(feedback)],
        interviewStopped: true,
        evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback, nextDifficulty: "same" },
        directive: "",
    };
}

// ═══════════════════════════════════════════════════════════════════
// ROUTING
// ═══════════════════════════════════════════════════════════════════
function routeEntryDeep(state) {
    if (state.userAnswer) return "orchestrator";
    if (!state.questionPlan || state.questionPlan.length === 0) return "planCurriculum";
    return "askQuestion";
}

function afterOrchestrator(state) {
    switch (state.dispatch) {
        case "evaluate":    return "evaluateAnswer";
        case "control":     return "controlFlow";
        case "backchannel": return "backchannel";
        case "converse":
        default:            return "converse";
    }
}

// Parallel fan-out after grading (mirrors classic afterGradeRoute)
function afterEvaluateDeep() {
    return ["adaptDifficulty", "prefetchContext"];
}

function afterMemoryDeep(state) {
    if (state.interviewStopped) return "generateReport";
    const wantsQuestion = state.dispatch === "evaluate" || (state.dispatch === "converse" && state.needsQuestion);
    if (!wantsQuestion) return END;
    // follow-ups don't increment questionsAsked, so they're allowed even at max
    if (state.questionsAsked >= state.maxQuestions && !state.followUpFlag) return "generateReport";
    return "askQuestion";
}

// ═══════════════════════════════════════════════════════════════════
// GRAPH
// ═══════════════════════════════════════════════════════════════════
export async function createDeepInterviewAgent(checkpointer) {
    if (checkpointer === undefined && process.env.SUPABASE_DB_URL) {
        checkpointer = PostgresSaver.fromConnString(process.env.SUPABASE_DB_URL);
        await checkpointer.setup();
        interviewLog.info('Deep agent: PostgresSaver checkpointer initialized');
    }

    const workflow = new StateGraph({ channels: deepInterviewStateChannels })
        // reused nodes (aliased to deep-agent names)
        .addNode("planCurriculum",  planCurriculumNode)
        .addNode("evaluateAnswer",  gradeAnswerNode)
        .addNode("backchannel",     backchannelNode)
        .addNode("prefetchContext", prefetchContextNode)
        .addNode("adaptDifficulty", adaptDifficultyNode)
        .addNode("memory",          updateSummaryNode)
        .addNode("askQuestion",     generateQuestionNode)
        .addNode("generateReport",  generateFinalReportNode)
        // new deep-agent nodes
        .addNode("orchestrator",    orchestratorNode)
        .addNode("converse",        converseNode)
        .addNode("controlFlow",     controlFlowNode)

        // entry
        .addConditionalEdges(START, routeEntryDeep, {
            planCurriculum: "planCurriculum",
            askQuestion:    "askQuestion",
            orchestrator:   "orchestrator",
        })
        .addEdge("planCurriculum", "askQuestion")
        .addEdge("askQuestion", END)

        // orchestrator → specialist
        .addConditionalEdges("orchestrator", afterOrchestrator, {
            evaluateAnswer: "evaluateAnswer",
            converse:       "converse",
            controlFlow:    "controlFlow",
            backchannel:    "backchannel",
        })
        .addEdge("backchannel", END)

        // evaluate → parallel enrich → memory
        .addConditionalEdges("evaluateAnswer", afterEvaluateDeep, {
            adaptDifficulty: "adaptDifficulty",
            prefetchContext: "prefetchContext",
        })
        .addEdge("adaptDifficulty", "memory")
        .addEdge("prefetchContext", "memory")

        // converse → memory (consolidate)
        .addEdge("converse", "memory")

        // memory → ask | report | end  (no pathMap — afterMemoryDeep may return END)
        .addConditionalEdges("memory", afterMemoryDeep)

        // control → report
        .addEdge("controlFlow", "generateReport")
        .addEdge("generateReport", END);

    const compileOpts = {};
    if (checkpointer) compileOpts.checkpointer = checkpointer;
    return workflow.compile(compileOpts);
}
