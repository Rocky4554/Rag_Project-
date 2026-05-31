# Interview Agent V2: Deep Agent Architecture Plan

---

## Context

The current LangGraph interview agent works as a deterministic state machine. The `routerNode` classifies intent and routes to either `gradeAnswerNode` (for real answers) or `handleEdgeCaseNode` (for everything else). The problem is that `handleEdgeCaseNode` is a 240-line God Node with 9 hardcoded branches that respond robotically:

- User goes off-topic → "Let's stay on topic" (ignores what they actually said)
- User is nervous → plays cached phrase "[take_your_time]" (no warmth, no context)
- User asks how many questions remain → returns a raw number (no encouragement)

A real human interviewer would **acknowledge what the candidate said**, **read the room** (are they nervous for the 2nd time? improving? struggling?), and **bring them back naturally**. This upgrade introduces a `lightweightOrchestratorNode` that reasons about the full behavioral context before handing off to a new `conversationManagerNode` — replacing the God Node with intelligent, context-aware responses.

**The happy path (answer → grade → next question) is completely unchanged.** All new overhead only applies to non-answer turns (~30% of turns).

---

## Current Architecture Problems

| Problem | Current Behavior | Goal |
|---|---|---|
| Off-topic response | "Let's stay on topic" | Acknowledge what they said, bridge back naturally |
| Nervous response | Cached "[take_your_time]" phrase | Reference past scores: "You scored 8 on the last question, just tell me what you know" |
| Pattern blindness | Treats 3rd nervousness same as 1st | Track pattern: escalate warmth progressively |
| Meta questions | Hardcoded number only | Warm answer: "2 left and you're averaging 7/10 — you're doing great!" |
| Confused candidate | Single rephrase attempt | Context-aware rephrase using orchestrator guidance on WHICH part confused them |
| No conversation memory | Each non-answer turn handled in isolation | Rolling `conversationPattern[]` tracks last 5 non-answer interactions |

---

## New Architecture Overview

```
START
  ↓ routeOnStart
  ├── planCurriculum → generateQuestion → END   (unchanged)
  └── routerNode
        ├── ANSWER          → gradeAnswer → [adaptDifficulty ‖ prefetchContext] → updateSummary → generateQuestion|report
        ├── THINKING/CUTOFF → backchannel → END
        ├── STOP/UNWELL     → conversationManager → generateFinalReport   (direct, no orchestrator)
        └── everything else → lightweightOrchestrator → conversationManager → END|generateQuestion|report
```

**Key design principle:** The orchestrator does NOT change routing. It writes an `orchestratorInstruction` string into state that `conversationManagerNode` reads. If the orchestrator times out (300ms hard limit), `conversationManager` still runs with a fallback instruction. No cascading failures.

---

## New Nodes

### `lightweightOrchestratorNode`

**Model:** `gemini-2.5-flash-lite` (direct, no fallback chain — timeout is the fallback)  
**Temperature:** 0.4 | **Max tokens:** 120 | **Hard timeout:** 300ms

**What it reads from state:**
- `intent`, `userAnswer`, `currentQuestion` — what just happened
- `chatHistory` (last 2 turns), `scores[]`, `questionsAsked/maxQuestions` — interview context
- `interviewMood`, `conversationPattern[]`, `struggleStreak`, `hintsGiven` — behavioral history

**What it writes to state:**
- `orchestratorInstruction` — 1-2 sentence instruction for conversationManager
- `interviewMood` — updated mood reading: `neutral|nervous|frustrated|engaged|lost`
- `conversationPattern` — appends current intent (reducer keeps last 5)
- `lastInteractionType` — current intent for next turn's orchestrator context

**Prompt structure:**
```
You are an interview session advisor. A candidate just said something that is NOT a direct answer.
Write ONE instruction for the interviewer that:
1. Acknowledges what the candidate ACTUALLY said (not just the intent category)
2. Calibrates tone based on the pattern (repeated nervousness = warmer; first-time off-topic = lighter touch)
3. Suggests how to bridge back to the interview

CONTEXT: intent=${intent}, progress=Q${questionsAsked}/${maxQuestions}, avg=${avgScore}/10,
          struggle_streak=${struggleStreak}, pattern=${conversationPattern}, hints_given=${hintsGiven}
CANDIDATE SAID: "${userAnswer}"
CURRENT QUESTION: "${currentQuestion}"
RECENT CONVERSATION: [last 2 turns]

OUTPUT JSON: { "instruction": "...", "mood": "...", "tone": "warm|encouraging|professional|playful" }
```

**Graceful degradation:** If Gemini call fails or times out → returns fallback instruction string — `conversationManager` still runs correctly.

---

### `conversationManagerNode`

Replaces `handleEdgeCaseNode`. Three execution tiers:

#### Tier 1 — No LLM (< 5ms): `break_request`, `gratitude`
Deterministic warm responses.
```
break_request → "Of course, take all the time you need. Just start speaking whenever you're ready."
gratitude     → "You're very welcome! Let's keep the momentum going."
```

#### Tier 2 — Fast LLM < 150ms (`gemini-2.5-flash-lite`, 80-100 tokens): `hint_request`, `meta`, `skip`
- **hint** — RAG-powered progressive hint (max 2 per question, hint 2 more specific than hint 1)
- **meta** — Uses exact numbers from state + orchestrator instruction: "2 left and averaging 7/10!"
- **skip** — One warm acknowledgment sentence then advances

#### Tier 3 — Rich LLM < 300ms (Groq `llama-3.3-70b`, 120 tokens, streaming): `irrelevant`, `nervous`, `confused`, `greeting`

**IRRELEVANT** — Acknowledge + bridge, never "stay on topic":
```
System: "You are a warm interviewer. The orchestrator says: ${orchestratorInstruction}
RULES: 1-2 sentences. Acknowledge what they SPECIFICALLY said. Bridge back naturally.
NEVER say 'Let's stay on topic'. Sound like a real person."
```
- `offTopicCount >= 3` → escalated firm redirect
- AI identity questions → instant fast-path: "I'm your AI interviewer! Anyway, let's get back to where we were."

**NERVOUS** — Reference actual past performance:
```
System: "A candidate is nervous. ${pastScores} Use this to ground them.
Don't say 'take your time' alone. Add something specific. 1 sentence max."
```
- `scores.length >= 1` → references actual past scores
- `scores.length === 0` → general encouragement only (no fabricated context)

**CONFUSED** — Orchestrator-guided rephrase. Returns updated `currentQuestion`.

**STOP / UNWELL** — Unchanged: cached TTS + `interviewStopped: true` (bypass orchestrator)

---

## New State Fields

Add to `interviewStateChannels` after `offTopicCount`:

```js
orchestratorInstruction: { value: (x, y) => y ?? x, default: () => "" },
interviewMood:           { value: (x, y) => y ?? x, default: () => "neutral" },
lastInteractionType:     { value: (x, y) => y ?? x, default: () => "" },

// Append reducer — keeps last 5 non-answer interaction types
conversationPattern: {
    value: (x, y) => {
        if (!y) return x;
        return [...(x || []), ...y].slice(-5);
    },
    default: () => []
},
```

**Total: 4 new fields.** All existing 32 fields unchanged.

---

## New Routing Functions

### `afterRouterRouteV2` (replaces `afterRouterRoute`)
```js
function afterRouterRouteV2(state) {
    if (state.intent === INTENTS.ANSWER || state.intent === INTENTS.NORMAL)
        return "gradeAnswer";
    if (state.intent === INTENTS.THINKING || state.intent === INTENTS.CUTOFF)
        return "backchannel";
    // Terminal intents — bypass orchestrator (unambiguous, no reasoning needed)
    if (state.intent === INTENTS.STOP || state.intent === INTENTS.UNWELL)
        return "conversationManager";
    // All other non-answer intents → orchestrator enriches first
    return "lightweightOrchestrator";
}
```

### `afterConversationManagerRoute` (replaces `afterEdgeCaseRoute`)
```js
function afterConversationManagerRoute(state) {
    if (state.interviewStopped) return "generateFinalReport";
    if (state.questionsAsked >= state.maxQuestions) return "generateFinalReport";
    const feedbackOnlyIntents = [
        INTENTS.CONFUSED, INTENTS.META, INTENTS.IRRELEVANT,
        INTENTS.BREAK, INTENTS.HINT, INTENTS.GRATITUDE, INTENTS.GREETING
    ];
    if (feedbackOnlyIntents.includes(state.intent)) return END;
    return "generateQuestion"; // SKIP, NERVOUS, UNWELL
}
```

---

## Graph Wiring Changes

In `createInterviewAgent`:

```js
// REMOVE:
.addNode("handleEdgeCase", handleEdgeCaseNode)

// ADD:
.addNode("lightweightOrchestrator", lightweightOrchestratorNode)
.addNode("conversationManager",     conversationManagerNode)

// UPDATE routerNode conditional edges:
.addConditionalEdges("routerNode", afterRouterRouteV2, {
    gradeAnswer:             "gradeAnswer",
    backchannel:             "backchannel",
    conversationManager:     "conversationManager",
    lightweightOrchestrator: "lightweightOrchestrator",
})

// ADD new edges:
.addEdge("lightweightOrchestrator", "conversationManager")
.addConditionalEdges("conversationManager", afterConversationManagerRoute)
```

All other edges are **unchanged**.

---

## Helper Functions to Add

Add near `buildCompactHistory` in `interviewAgent.js`:

```js
function makeGeminiMini(temperature = 0.5, maxTokens = 90) {
    return new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: "gemini-2.5-flash-lite",
        temperature,
        maxOutputTokens: maxTokens,
    });
}

function buildConversationManagerReturn(state, feedbackText, interviewStopped = false, extras = {}) {
    return {
        chatHistory: [
            ...state.chatHistory,
            new HumanMessage(state.userAnswer),
            new AIMessage(feedbackText)
        ],
        interviewStopped,
        evaluation: { score: 0, accuracy: 0, clarity: 0, depth: 0, feedback: feedbackText, nextDifficulty: "same" },
        orchestratorInstruction: "", // always reset after consumption
        ...extras
    };
}
```

---

## Worker.js Changes (minor)

**File:** `agents/interview/worker.js`

Update log at line ~116:
```js
agentLog.info({ sessionId, intent: result.intent, done: result.done,
    mood: result.interviewMood,
    pattern: result.conversationPattern,
}, 'LangGraph result');
```

Update comment at line ~193:
```js
// ── Conversation manager handled — speak feedback only ──
// conversationManagerNode already built the full response in evaluation.feedback
```

**No changes to:** `sessionBridge.js`, `routes/interview.js`, or any frontend files.

---

## `generateQuestionNode` Enhancement (bonus)

When rephrasing for nervous/irrelevant turns, inject orchestrator tone guidance:

```js
const repeatGuidance = effectiveRepeat && state.orchestratorInstruction
    ? `\nTONE GUIDANCE: ${state.orchestratorInstruction}`
    : "";
// Append ${repeatGuidance} to the systemPrompt
```

---

## Implementation Order

| Step | Task | File | Est. Time |
|---|---|---|---|
| 1 | Add 4 new state channels | `interviewAgent.js` | 15 min |
| 2 | Add `makeGeminiMini` + `buildConversationManagerReturn` helpers | `interviewAgent.js` | 10 min |
| 3 | Implement `lightweightOrchestratorNode` with 300ms timeout | `interviewAgent.js` | 45 min |
| 4 | Implement `conversationManagerNode` (all 3 tiers) | `interviewAgent.js` | 60 min |
| 5 | Add `afterRouterRouteV2` + `afterConversationManagerRoute` | `interviewAgent.js` | 20 min |
| 6 | Update graph wiring in `createInterviewAgent` | `interviewAgent.js` | 20 min |
| 7 | Inject `repeatGuidance` into `generateQuestionNode` | `interviewAgent.js` | 15 min |
| 8 | Update logs/comments in `worker.js` | `worker.js` | 10 min |
| 9 | Integration test all 14 intents | manual | 60 min |
| **Total** | | | **~4 hours** |

> **Keep `handleEdgeCaseNode` as `handleEdgeCaseNode_DEPRECATED`** until 1 week of production stability. Rollback = swap `addNode` reference.

---

## Files to Modify

| File | What Changes |
|---|---|
| `lib/interview/interviewAgent.js` | 4 state fields, 2 helpers, 2 new nodes, 2 routing functions, graph wiring, `generateQuestionNode` bonus |
| `agents/interview/worker.js` | Log + comment update only |

---

## Reused Functions (do not rewrite)

- `buildCompactHistory(chatHistory, n)` — `interviewAgent.js` ~line 344
- `getContext(sessionId, query, k)` — `interviewAgent.js` ~line 180
- `makeLLM(temp, tokens, streaming)` — `interviewAgent.js` ~line 293
- `checkTTSCache(phraseKey)` — `interviewAgent.js` ~line 80
- `getSocket(sessionId)` — `interviewAgent.js` ~line 203
- `INTENTS` object — `interviewAgent.js` ~line 269

---

## Verification

### Log signals after deployment

| Signal | What to look for |
|---|---|
| Orchestrator ran | `'Orchestrator analyzing non-answer turn'` with `intent` + `mood` |
| Orchestrator timed out | `'Orchestrator timed out — using fallback instruction'` — must NOT cause 500 |
| Instruction consumed | `'ConversationManager handling turn'` with non-empty `instruction` |
| Pattern tracking | After 3+ non-answer turns: `conversationPattern` shows last 5 intents |
| No robotic phrases | `grep "stay on topic"` in logs → zero results |
| Mood persistence | `interviewMood: "nervous"` persists across turns for repeatedly nervous candidates |

### Regression check — grade path unchanged

- `afterRouterRouteV2("answer_attempt")` → `"gradeAnswer"` ✅
- None of the 4 new state fields written by gradeAnswer/adapt/prefetch/summary/generateQuestion nodes

### Manual test matrix

```
1.  Answer normally        → grade path, no orchestrator in logs
2.  "hmm let me think"    → backchannel, no orchestrator
3.  "stop the interview"  → conversationManager direct (no orchestrator)
4.  "what time is it?"    → orchestrator → IRRELEVANT → acknowledge + bridge back
5.  "I'm so nervous"      → orchestrator → NERVOUS → references past scores
6.  "how many questions?" → orchestrator → META → warm count + avg score
7.  "give me a hint"      → orchestrator → HINT → progressive clue
8.  "can we take a break?"→ orchestrator → BREAK → warm pause response
9.  "I don't know"        → SKIP → warm skip → generateQuestion advances
10. "what do you mean?"   → orchestrator → CONFUSED → smart rephrase
```
