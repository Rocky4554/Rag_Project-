# Interview Agent — Complete Technical Reference

> This document covers the entire interview agent system end-to-end: from the moment a user clicks "Start Interview" to the final report being saved. Every component, every LLM call, every WebSocket message, every state field.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [File Map](#3-file-map)
4. [Phase 1 — PDF Upload (prerequisite)](#4-phase-1--pdf-upload-prerequisite)
5. [Phase 2 — Interview Start (`POST /api/interview/start`)](#5-phase-2--interview-start)
6. [Phase 3 — LiveKit Agent Startup](#6-phase-3--livekit-agent-startup)
7. [Phase 4 — First Question (LangGraph Invocation #1)](#7-phase-4--first-question-langgraph-invocation-1)
8. [Phase 5 — Real-Time Voice Turn (per answer)](#8-phase-5--real-time-voice-turn-per-answer)
9. [The LangGraph State Machine — All Nodes](#9-the-langgraph-state-machine--all-nodes)
10. [LLM Factory & Fallback Chain](#10-llm-factory--fallback-chain)
11. [Deepgram STT — How Speech Becomes Text](#11-deepgram-stt--how-speech-becomes-text)
12. [Polly TTS — How Text Becomes Speech](#12-polly-tts--how-text-becomes-speech)
13. [AudioPublisher — How PCM Reaches the Browser](#13-audiopublisher--how-pcm-reaches-the-browser)
14. [TTS Phrase Cache System](#14-tts-phrase-cache-system)
15. [Socket.io Events Reference](#15-socketio-events-reference)
16. [Full LangGraph State — All 24 Fields](#16-full-langgraph-state--all-24-fields)
17. [Phase 6 — Interview Completion & Final Report](#17-phase-6--interview-completion--final-report)
18. [Phase 7 — User Profile Update (Cross-Session Memory)](#18-phase-7--user-profile-update-cross-session-memory)
19. [Complete Flow Diagram](#19-complete-flow-diagram)
20. [Timing & Performance](#20-timing--performance)

---

## 1. System Overview

The interview agent conducts a fully automated, real-time **voice interview** about a PDF the user uploaded. The system:

- Listens to the user's microphone via **LiveKit WebRTC**
- Transcribes speech in real-time using **Deepgram nova-3 STT**
- Evaluates answers and generates questions using **LangGraph** (a stateful graph) with **Groq / LLaMA 3.3 70B** as the LLM
- Speaks responses back via **AWS Polly neural TTS** → **LiveKit WebRTC**
- Tracks difficulty, detects intent, scores answers per topic
- Adapts question difficulty in real-time based on performance
- Generates a final markdown debrief report
- Saves results to Supabase and updates the user's long-term profile

The architecture is **pipeline-free** from the user's perspective — they simply speak and hear responses. Internally it's:

```
Browser mic → LiveKit → Deepgram WS → LangGraph (Groq LLM) → Polly TTS → LiveKit → Browser speaker
```

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| WebRTC transport | LiveKit (cloud or self-hosted) | Audio in/out over the internet |
| Server LiveKit SDK | `@livekit/rtc-node` | Node.js agent joins room, reads mic, publishes AI audio |
| Speech-to-Text | Deepgram nova-3 | Converts user voice to text in real-time |
| Text-to-Speech | AWS Polly neural | Converts AI text to PCM audio |
| State machine | LangGraph (`@langchain/langgraph`) | Manages interview flow graph with checkpointing |
| LLM | Groq (llama-3.3-70b-versatile) → OpenRouter → Gemini | Evaluates answers, generates questions |
| Checkpointing | LangGraph PostgresSaver → Supabase Postgres | Persists state between turns |
| RAG retrieval | Qdrant vector DB + Gemini embeddings | Context for questions and evaluations |
| Real-time comms | Socket.io | Pushes transcript, score, and state events to browser |
| Persistence | Supabase Postgres | Saves interview results, user profiles |

---

## 3. File Map

```
agents/
  interview/
    worker.js          InterviewAgentWorker class — orchestrates STT, TTS, LiveKit, LangGraph
    sessionBridge.js   Bridges worker ↔ LangGraph state machine
    stt.js             DeepgramSTT — WebSocket connection to Deepgram
    tts.js             generatePCM / generatePCMPipelined — AWS Polly calls
  shared/
    audioPublisher.js  AudioPublisher — chunks PCM into 10ms frames for LiveKit

lib/
  interview/
    interviewAgent.js  LangGraph graph definition + all 8 nodes + routing logic
    profileUpdater.js  Cross-session user profile: save/load weak/strong areas
    quizGenerator.js   (separate feature, not used by interview agent)
    summarizer.js      (separate feature, not used by interview agent)
  llm.js               LLM factory with Groq → OpenRouter → Gemini fallback
  pipeline/
    vectorStore.js     Qdrant vector store operations
    rag.js             RAG query helper
  sessionRestore.js    Restores session from Supabase+Qdrant if not in memory

routes/
  interview.js         POST /api/interview/start — entry point

assets/tts/            Pre-recorded MP3 phrase files (14 files)
  great_answer.mp3
  good_effort.mp3
  take_your_time.mp3
  ... (see Section 14)
```

---

## 4. Phase 1 — PDF Upload (prerequisite)

Before an interview can start, a PDF must be uploaded and processed. This creates the `vectorStore` that the interview agent uses for RAG retrieval.

**Route:** `POST /api/upload` (handled by `routes/upload.js`)

**Steps:**
1. `multer` receives the file into memory (`memoryStorage()` — no disk file created)
2. A unique `sessionId` is generated: `randomBytes(12).toString('hex') + '-' + Date.now()`
3. The buffer is written to a temp file in `os.tmpdir()` (e.g. `/tmp/rag-abc123-1234567890.pdf`)
4. `extractTextFromPDF(tmpPath)` → LangChain `PDFLoader` → returns array of Document objects (one per page)
5. The temp file is immediately deleted (fire-and-forget `fs.unlink`)
6. `splitText(docs)` → 900-character chunks with 150-character overlap
7. `storeDocuments(chunks, sessionId)` → Gemini `gemini-embedding-001` embeds each chunk → stored in Qdrant collection named `sessionId`
8. `sessionCache[sessionId] = { vectorStore, docs, chatHistory: [], originalName }` — in-memory cache
9. If authenticated user: saves document metadata to Supabase `documents` table
10. Response: `{ message, sessionId }`

The `sessionId` is the Qdrant collection name, the LangGraph thread ID, and the LiveKit room name — it ties everything together.

---

## 5. Phase 2 — Interview Start

**Route:** `POST /api/interview/start`
**File:** `routes/interview.js`
**Auth:** Optional (features like profile memory require auth)

### Request body
```json
{ "sessionId": "abc123-1234567890", "maxQuestions": 5 }
```

### What happens, step by step:

**Step 1 — Session lookup**
```js
const session = await ensureSession(sessionCache, sessionId);
```
Tries the in-memory `sessionCache` first. If missing (server restarted), calls `lib/sessionRestore.js` which queries Supabase for the document record and re-instantiates the Qdrant vector store. Throws `404` if not found.

**Step 2 — Register vector store**
```js
registerVectorStore(sessionId, session.vectorStore);
```
Stores the vector store in a module-level `Map` inside `interviewAgent.js` so all LangGraph nodes can call `getContext(sessionId, query)` without passing it through state.

**Step 3 — Load user profile (if authenticated)**
```js
const profileCtx = await getUserProfileContext(req.user.id);
```
Fetches the user's past interview history from Supabase `user_profiles` table. Returns a formatted string like:
```
[USER HISTORY: 3 past interview(s), avg score: 6.8/10]
Strong areas: React hooks, async/await
Weak areas (focus more here): database indexing, closures
Performance summary: Improving across sessions, avg went from 5.5 to 6.8.
```
This is injected into the LangGraph initial state and used by `planCurriculumNode` to bias topics toward weak areas, and by `generateQuestionNode` as background context.

**Step 4 — Build initial state**
```js
const initialState = {
    sessionId,
    maxQuestions: 5,
    difficultyLevel: "medium",    // starting difficulty
    chatHistory: [],
    questionsAsked: 0,
    topicsUsed: [],
    userProfileContext,           // from Step 3
    candidateName: "Raunak",      // from req.user.name or "there"
    timeGreeting: "Good morning", // based on server hour
};
```

**Step 5 — LangGraph config**
```js
const config = { configurable: { thread_id: sessionId } };
```
The `thread_id` is used by:
- **PostgresSaver**: checkpoints state after every node so the interview survives server restarts
- **LangSmith**: groups all LLM traces for this interview under one thread

**Step 6 — Create agent worker**
```js
const agent = new InterviewAgentWorker(sessionId, sessionCache, interviewAgent, io);
activeAgents.set(sessionId, agent);
```
Stores in the `activeAgents` Map so it can be stopped later if user navigates away.

**Step 7 — Register completion callback**
```js
session._onInterviewComplete = async (finalState) => { ... save to DB ... };
```
This callback is stored on the session object. When `handleUserTurn()` detects the interview is done, it calls this to save results to Supabase and trigger profile update.

**Step 8 — CRITICAL: Run LangGraph + agent.start() in PARALLEL**
```js
const [resultState] = await Promise.all([
    interviewAgent.invoke(initialState, config),  // LangGraph: plan + first question
    agent.start()                                  // LiveKit: connect + publish audio track
]);
```
Both operations run concurrently. LangGraph (~800ms LLM call) and LiveKit setup (~200-400ms) overlap, saving ~400ms of startup latency.

**Step 9 — Wait for client audio ready**
```js
await new Promise((resolve) => {
    clientReadyResolvers.set(sessionId, resolve);
    setTimeout(resolve, 2000);  // 2s timeout
});
```
The browser emits `client_audio_ready` via Socket.io once its LiveKit room connection is established and the browser mic is active. Server waits up to 2 seconds before speaking anyway.

**Step 10 — Speak first question**
```js
agent.speak("Hello! Welcome to your AI voice interview. " + uniquePart);
```
`uniquePart` = first question text with `[interview_intro]` and other TTS tags stripped (but phrase text added back). This goes through the full TTS pipeline.

**Step 11 — HTTP response**
```json
{ "questionNumber": 1, "difficulty": "medium", "agentStarted": true }
```
The browser now knows the interview is live and listens for Socket.io events.

---

## 6. Phase 3 — LiveKit Agent Startup

**File:** `agents/interview/worker.js` — `start()` method

This runs in parallel with the LangGraph first invocation (see Step 8 above).

```
[1/4] Starting Deepgram STT (nova-3) — non-blocking WebSocket fired
[2/4] Generating LiveKit token (+Xms)
[3/4] Connecting to LiveKit room (+Xms)
[4/4] Publishing AI audio track (+Xms, totalMs)
```

### Step-by-step

**[1/4] Deepgram STT starts**
```js
this.stt.start();   // non-blocking — WebSocket to Deepgram fires in background
```
`DeepgramSTT._connect()` creates a WebSocket to `wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=48000&channels=1&model=nova-3&...`

Parameters sent:
- `encoding=linear16` — raw 16-bit PCM (what LiveKit sends)
- `sample_rate=48000` — LiveKit default browser mic output
- `model=nova-3` — Deepgram's best model
- `smart_format=true` — auto-punctuation, numbers as digits
- `interim_results=true` — get partial transcripts for display
- `endpointing=300` — treat 300ms of silence as end-of-speech

**[2/4] LiveKit token generated**
```js
const token = await this.generateToken();
```
Uses `AccessToken` from `livekit-server-sdk`. Creates a JWT granting:
- Identity: `"ai-interviewer"`
- Room: `sessionId`
- Permissions: `canPublish=true`, `canSubscribe=true`

**[3/4] Connect to LiveKit room**
```js
await this.room.connect(process.env.LIVEKIT_URL, token);
```
The server agent joins the same LiveKit room as the browser. Both are now in the same room.

**[4/4] Publish AI audio track**
```js
const track = LocalAudioTrack.createAudioTrack("ai-interviewer-audio", this.audioPublisher.source);
await this.room.localParticipant.publishTrack(track, {
    source: TrackSource.SOURCE_MICROPHONE
});
```
`this.audioPublisher.source` is a `LiveKit AudioSource` object. When the agent pushes PCM frames to this source, they travel via WebRTC to the browser and play through the speaker.

**`setupRoomEvents()` registers two event handlers:**

```js
// 1. When browser joins and publishes mic track:
room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === TrackKind.KIND_AUDIO) {
        this.listenToUserAudio(track);  // start piping audio to Deepgram
    }
});

// 2. When LiveKit disconnects:
room.on(RoomEvent.Disconnected, () => {
    this.stop();
});
```

**`listenToUserAudio(track)`** — the real-time audio pipeline:
```js
const audioStream = new AudioStream(track);  // async iterable of AudioFrames
for await (const frame of audioStream) {
    if (!processingTurn && !audioPublisher.isSpeaking) {
        stt.pushAudio(frame.data);  // forward Int16Array PCM to Deepgram WebSocket
    }
}
```
- Audio is NOT forwarded while the agent is speaking (to avoid the agent hearing itself)
- Audio is NOT forwarded while a turn is being processed (LangGraph running)

**`stt.on("transcript", text)` handler:**
```js
this.stt.on("transcript", async (text) => {
    if (!isActive || processingTurn) return;
    if (audioPublisher.isSpeaking) return;  // basic VAD
    await handleUserTurn(text);
});
```

---

## 7. Phase 4 — First Question (LangGraph Invocation #1)

**File:** `lib/interview/interviewAgent.js`

The `interviewAgent.invoke(initialState, config)` call (running in parallel with `agent.start()`) traverses the graph:

### Routing entry: `routeOnStart(state)`
```js
function routeOnStart(state) {
    if (state.userAnswer) return "processAnswer";            // has answer → evaluate
    if (!state.questionPlan?.length) return "planCurriculum"; // no plan → plan first
    return "generateQuestion";                               // has plan, no answer → ask
}
```
First call: `userAnswer=""`, `questionPlan=[]` → routes to **`planCurriculum`**

---

### Node 0: `planCurriculumNode`

**Purpose:** Plan all N question topics upfront in one LLM call so every subsequent question is pre-targeted.

**What it does:**
1. `getContext(sessionId, "main topics key concepts overview summary", 5)` — retrieves 5 Qdrant chunks
2. If user has profile history: adds "focus on weak areas" instruction to the prompt
3. **LLM call:** `makeLLM(temperature=0.3, maxTokens=350)`

**LLM prompt (condensed):**
```
You are planning a technical interview curriculum based on a document.

DOCUMENT EXCERPTS: [700 chars from Qdrant]
[CANDIDATE HISTORY if exists: weak areas, strong areas, past score]

Create exactly 5 interview question slots in easy → hard progression.
Return ONLY a valid JSON array:
[
  { "topic": "...", "difficulty": "easy", "angle": "definition" },
  { "topic": "...", "difficulty": "medium", "angle": "how it works" },
  ...
]
Rules: Topics MUST come from the document. Each topic distinct. Prioritize weak areas.
```

**LLM response (example):**
```json
[
  { "topic": "React hooks", "difficulty": "easy", "angle": "definition and purpose" },
  { "topic": "useEffect", "difficulty": "medium", "angle": "dependency array behavior" },
  { "topic": "custom hooks", "difficulty": "medium", "angle": "how to build and share" },
  { "topic": "React context", "difficulty": "hard", "angle": "performance tradeoffs" },
  { "topic": "memoization", "difficulty": "hard", "angle": "useMemo vs useCallback" }
]
```

**Returns:** `{ questionPlan: [...] }`

**Fallback:** If JSON parse fails, generates default plan: easy/easy/medium/medium/hard topics labeled "core concepts".

---

### Routes to: `generateQuestionNode` (Path B, Q1)

(See full node description in Section 9)

For the first question:
- Reads `questionPlan[0]` (e.g. `{ topic: "React hooks", difficulty: "easy", angle: "definition" }`)
- `tagInstruction` = `"Start with [interview_intro] then say 'Good morning, Raunak! Let's get started.' then ask the first question."`
- LLM generates: `"[interview_intro] Good morning, Raunak! Let's get started. Can you explain what React hooks are and why they were introduced?"`
- Tokens streamed via `socket.emit("ai_stream", { token, type: "question" })` if socket connected
- Returns `{ currentQuestion: "...", questionsAsked: 1, topicTag: "React hooks", ... }`

**→ END** (graph stops here for the first invocation)

The `resultState.currentQuestion` is passed to `agent.speak()` in Phase 2 Step 10.

---

## 8. Phase 5 — Real-Time Voice Turn (per answer)

This is the repeating loop that runs for every question the user answers.

### 8.1 Audio Path: Browser → Deepgram

```
Browser mic (48kHz PCM, 16-bit mono)
  → LiveKit WebRTC (encrypted, low latency)
  → LiveKit server
  → Server: AudioStream iterator yields AudioFrame objects
  → stt.pushAudio(frame.data)  // Int16Array sent as binary over WebSocket
  → Deepgram nova-3 processes in real-time
  → Emits interim Results as user speaks
  → On speech_final (300ms silence): concatenates final segments → emit("transcript", text)
```

### 8.2 `handleUserTurn(transcript)` — main turn handler

**File:** `agents/interview/worker.js`

```
1. Log:  🎤 User said "..." (N words)
2. Emit: socket ai_state { state: "thinking", text: "Evaluating your answer..." }
3. Emit: socket transcript_final { role: "user", text: transcript }
4. Call: result = await sessionBridge.processUserTranscript(transcript)
         → agentWorkflow.invoke({ userAnswer: transcript }, config)
         → [LangGraph runs — see Section 9]
5. Log:  LangGraph turn result { intent, quality, score, topic, difficulty, nextDifficulty, questionNum }
6. If done:
     Emit: socket interview_done { report: finalReport }
     Call: session._onInterviewComplete(result)  → save to DB
     Play: "Thank you for your time. The interview is now complete."
     Call: setTimeout(stop, 2000)
7. If not done:
     Emit: socket transcript_final { role: "ai", text: nextQuestion }
     Emit: socket ai_state { state: "speaking" }
     Parse TTS tags from feedback + question
     Log:  TTS phrase cache hit (for each [tag])
     Either:
       A) feedbackText + questionText → parallel TTS (feedback first + question in background)
       B) combined text → pipelined TTS (all sentences concurrently)
     Play audio via audioPublisher.pushPCM()
     Emit: socket ai_state { state: "listening" }
```

### 8.3 TTS Tag Parsing (`parseTTSResponse`)

When LangGraph returns `evaluation.feedback = "[good_effort] That's a good start, but you missed the dependency array."`:

```js
parseTTSResponse("[good_effort] That's a good start...")
// Returns:
{
    phraseKeys: ["good_effort"],          // maps to assets/tts/good_effort.mp3
    uniquePart: "That's a good start, but you missed the dependency array."
}
```

In `worker.js`, the phraseKeys map to `PHRASE_TEXT`:
```js
PHRASE_TEXT["good_effort"] = "Good effort."
```
So the agent speaks: `"Good effort. That's a good start, but you missed the dependency array."` — the phrase "Good effort." comes from a pre-written text (TTS'd inline via Polly), the rest is the LLM's unique feedback.

---

## 9. The LangGraph State Machine — All Nodes

The graph is compiled once at server startup (`createInterviewAgent()`) and reused for all sessions. State is persisted per `thread_id` (= `sessionId`) via `PostgresSaver`.

### Graph Structure

```
START
  │
  ├─[no answer, no plan]──→ planCurriculum ──→ generateQuestion ──→ END
  │
  ├─[no answer, has plan]──────────────────────────────────────────────────────────────┐
  │                                                                                     │
  └─[has answer]──→ processAnswer                                                      │
                         │                                                             │
                   [normal]──→ adaptDifficulty ──┐                                    │
                         │                       ├──→ updateSummary ──→ generateQuestion ──┘
                   [normal]──→ prefetchContext ──┘          │
                         │                                   └─[all questions done]──→ generateFinalReport ──→ END
                   [non-normal]──→ handleEdgeCase
                                        │
                                  [stopped]──→ generateFinalReport ──→ END
                                        │
                                  [continue]──→ generateQuestion ──→ END
```

---

### Node 0: `planCurriculumNode`

**When:** First invocation only (no `questionPlan` yet)
**LLM:** 1 call — temp=0.3, maxTokens=350
**Input state used:** `sessionId`, `maxQuestions`, `userProfileContext`
**RAG:** Yes — 5 chunks, query: "main topics key concepts overview summary"

**What it produces:** `questionPlan[]` — an array of `{ topic, difficulty, angle }` objects covering all N questions. Example:
```json
[
  { "topic": "React hooks",   "difficulty": "easy",   "angle": "definition and purpose" },
  { "topic": "useEffect",     "difficulty": "medium", "angle": "dependency array behavior" },
  { "topic": "custom hooks",  "difficulty": "medium", "angle": "building reusable logic" },
  { "topic": "React context", "difficulty": "hard",   "angle": "performance tradeoffs" },
  { "topic": "memoization",   "difficulty": "hard",   "angle": "useMemo vs useCallback" }
]
```

If `userProfileContext` exists, the prompt says: "Prioritize the candidate's WEAK AREAS when choosing topics."

**Log output:**
```
Planning curriculum from document  { sessionId, maxQuestions: 5 }
Curriculum ready  { plan: "React hooks(easy) -> useEffect(medium) -> ..." }
```

---

### Node 1: `processAnswerNode`

**When:** Every time `userAnswer` is set (every turn after turn 1)
**LLM:** 1 call — temp=0.1, maxTokens=300 (or 0 calls if silence detected)
**Input state used:** `userAnswer`, `currentQuestion`, `sessionId`, `topicTag`, `followUpAsked`, `struggleStreak`, `chatHistory`
**RAG:** Yes — 3 chunks, query: `currentQuestion`

**Silence detection (no LLM):**
If the answer is < 5 words, returns immediately:
```js
{
    intent: "nervous",
    answerQuality: "skipped",
    evaluation: { score: 0, feedback: "[take_your_time]", nextDifficulty: "same" }
}
```

**LLM call — combined intent detection + evaluation:**

The prompt gives the LLM:
- The question text
- The candidate's answer (with note: "may have speech recognition errors")
- 700 chars of reference material from Qdrant

Task 1 — Detect intent:
- `"normal"` — any genuine attempt to answer (default, very lenient)
- `"skip"` — user says "I don't know", "skip", "next question"
- `"stop"` — user says "stop", "end interview", "quit"
- `"nervous"` — "I'm blanking", "I can't think right now"
- `"unwell"` — "I'm sick", "I need a break"
- `"irrelevant"` — completely off-topic (rare, strict)

Task 2 — Score the answer (only if intent=normal):
- `score`: 1-10 overall
- `accuracy`: 1-10 factual correctness
- `clarity`: 1-10 how clearly expressed
- `depth`: 1-10 how thorough
- `answerQuality`: "correct" (7-10) | "partial" (4-6) | "wrong" (1-3)
- `feedback`: one natural spoken sentence, warm but honest
- `nextDifficulty`: "easier" if score<5, "same" if 5-7, "harder" if >=8
- `topicTag`: 2-4 word label for the topic (e.g. "React hooks")
- `keyConceptMissed`: the specific thing they got wrong (drives follow-up)

**After the LLM response:**

1. **TTS quality tag is added to feedback:**
   - score ≥ 7 → `"[great_answer] That was a solid explanation."`
   - score 4-6 → `"[good_effort] You touched on the key idea, but missed the..."`
   - score ≤ 3 → `"[interesting] Let me explain — closures work by..."`

2. **Follow-up decision:**
   ```js
   shouldFollowUp = (intent === "normal") && (answerQuality === "partial") && !followUpAsked
   ```
   If true: sets `followUpFlag=true` which tells `generateQuestionNode` to dig deeper instead of advancing.

3. **Struggle streak:** consecutive wrong answers increments `struggleStreak`. If ≥ 2, `generateQuestionNode` uses a gentler, simpler tone.

4. **Per-topic scores:** `topicScores["React hooks"] = [6, 8, 7]` — all scores for this topic.

5. **Socket emit:** `ai_feedback { feedback, score, answerQuality, topicTag, followUpcoming }`

**Log output:**
```
Processing answer  { questionNum: 2, answerLength: 87 }
LLM request started  { provider: "groq", model: "llama-3.3-70b-versatile" }
LLM response received  { durationMs: 850, promptTokens: 312, completionTokens: 68, totalTokens: 380 }
Answer processed  { intent: "normal", quality: "partial", score: 6, nextDifficulty: "same", followUp: true, topic: "React hooks" }
```

---

### Node 2: `handleEdgeCaseNode`

**When:** `intent !== "normal"` (stop, skip, nervous, unwell, irrelevant)
**LLM:** 0 or 1 call

For known intents, uses a pre-recorded TTS phrase:

| Intent | TTS key | Phrase |
|---|---|---|
| `stop` | `interview_stopped` | "The interview has been stopped." |
| `skip` | `thats_okay` | "That's okay." |
| `nervous` | `take_your_time` | "Take your time." |
| `unwell` | `no_worries` | "No worries." |
| `irrelevant` | `out_of_context` | "Let's stay on topic." |

For "stop" and "unwell": also emits `play_tts` socket event with the MP3 base64 for browser-side playback.

If `intent === "stop"` OR it's the last question: sets `interviewStopped=true` → routes to `generateFinalReport`.

For unexpected intents not in the list: **1 LLM call** (streaming) generates a brief redirection sentence.

**Log output:**
```
Edge case handled  { intent: "nervous", cachedTTS: "take_your_time", shouldEndNow: false }
```

---

### Node 3: `prefetchContextNode`

**When:** After `processAnswer` when intent=normal, runs in PARALLEL with `adaptDifficulty`
**LLM:** None — pure RAG
**Purpose:** Pre-load the Qdrant context for the NEXT question topic before `generateQuestionNode` needs it

The node reads ahead in `questionPlan`:
```js
const planIdx = followUpFlag
    ? questionsAsked - 1   // follow-up: same curriculum slot
    : questionsAsked;       // next: advance one slot
const query = planItem ? `${planItem.topic} ${planItem.angle}` : "core concepts";
```

By prefetching here (in parallel with `adaptDifficulty`), `generateQuestionNode` can use `state.prefetchedContext` directly and skip the Qdrant call — saving ~150ms per turn.

**Log output:**
```
Prefetching context  { query: "useEffect dependency array behavior", planIdx: 1 }
```

---

### Node 4: `adaptDifficultyNode`

**When:** After `processAnswer` when intent=normal, runs in PARALLEL with `prefetchContext`
**LLM:** None — pure logic

```js
const levels = ["easy", "medium", "hard"];
if (nextDifficulty === "harder" && idx < 2) idx++;
else if (nextDifficulty === "easier" && idx > 0) idx--;
```

The `nextDifficulty` signal comes from the LLM in `processAnswerNode`:
- Score ≥ 8: "harder" → moves up one level
- Score < 5: "easier" → moves down one level
- Score 5-7: "same" → no change

**Log output:**
```
Difficulty adjusted  { from: "medium", to: "hard" }
```

---

### Node 4b: `updateSummaryNode`

**When:** After both `adaptDifficulty` AND `prefetchContext` complete (fan-in)
**LLM:** None — deterministic

Builds a compact summary of all scores so far:
```
"✓ React hooks: 8/10 | ~ useEffect: 6/10 | ✗ closures: 3/10"
```
(✓ = ≥7, ~ = 4-6, ✗ = <4)

Also trims `chatHistory` to the last 6 messages (3 Q&A pairs) to keep LLM context from growing unboundedly.

The summary is injected into `generateQuestionNode`'s prompt as `INTERVIEW PROGRESS:` so the LLM knows how the candidate is doing without seeing the full history.

---

### Node 5: `generateQuestionNode`

**When:** After `updateSummaryNode` (or directly after `planCurriculum` for Q1)
**LLM:** 1 call — temp=0.85, maxTokens=100 (streaming if socket connected)

**Two paths:**

#### Path A — Follow-up question (`followUpFlag=true`)
Uses `keyConceptMissed` and `topicTag` from the previous answer to probe deeper.

The prompt structure:
```
ORIGINAL QUESTION: [question that was asked]
CANDIDATE'S ANSWER: "[their answer]"
KEY CONCEPT THEY MISSED: [e.g. "closure scope"]
TOPIC: React hooks
SOURCE MATERIAL: [Qdrant context]

RULES:
- Ask ONE follow-up targeting the specific gap
- Stay on SAME topic — do NOT introduce new concepts
- Guide toward missed concept without giving it away
- 1-2 sentences max, warm and curious
```

`questionsAsked` is NOT incremented — this is a sub-question within the same curriculum slot.

#### Path B — Next planned question (`followUpFlag=false`)
Reads `questionPlan[questionsAsked]` for topic + difficulty + angle.

Special logic:
- **Repeat detection:** If previous intent was nervous or irrelevant, rephrase the same question (up to 2 times, then force-advance)
- **Supportive tone:** If `struggleStreak ≥ 2`, uses simpler/gentler phrasing
- **TTS tags added by the LLM:**
  - Q1: `[interview_intro]` + greeting
  - Last Q: `[final_question]`
  - Other: `[next_question]`

The prompt:
```
You are a senior engineer interviewing a candidate. Ask question #3 of 5.

TOPIC TO COVER: useEffect
ANGLE: dependency array behavior
DIFFICULTY: medium
- easy   → basic definitions, simple facts
- medium → understanding, cause-and-effect, how it works
- hard   → analysis, tradeoffs, edge cases, real-world design

SOURCE MATERIAL: [Qdrant context for useEffect]
CANDIDATE BACKGROUND: [user profile context]
INTERVIEW PROGRESS: ✓ React hooks: 8/10 | ~ closures: 5/10
RECENT CONVERSATION: [last 2 Q&A pairs]
TOPICS ALREADY COVERED: React hooks; closures

RULES:
- Start with [next_question] then ask the question.
- ONE question only, no preamble, 1-2 sentences max
- Target the topic and angle above
```

**Streaming:** If a socket is connected, tokens are streamed live:
```js
socket.emit("ai_stream", { token: "What", type: "question" })
socket.emit("ai_stream", { token: " happens", type: "question" })
...
socket.emit("ai_stream_end", { type: "question" })
```
This drives the animated text in the browser UI.

**Log output:**
```
Generating question  { questionNum: 3, topic: "useEffect", difficulty: "hard", repeat: false, supportiveTone: false }
LLM response received  { durationMs: 720, totalTokens: 95 }
Question generated  { questionNum: 3, question: "[next_question] Can you explain what happens when...", topic: "useEffect", difficulty: "hard" }
```

→ **END** (graph stops, returns to `sessionBridge.processUserTranscript()`)

---

### Node 6: `generateFinalReportNode`

**When:** `interviewStopped=true` OR `questionsAsked >= maxQuestions`
**LLM:** 1 call — temp=0.3, maxTokens=400

Pre-computes all stats before the LLM call:
```js
avg   = "6.8"   // mean of all scores
best  = 9       // max score
worst = 4       // min score
trend = "improving"  // last score vs first score
grade = "B"     // based on avg thresholds: A≥8.5, B≥7, C≥5, D≥3, F<3

// Per-topic breakdown:
"✅ React hooks: 8.3/10"
"⚠️ useEffect: 6.0/10"
"❌ closures: 3.5/10"
```

The LLM generates a markdown debrief:
```markdown
## Overall Grade
B — Solid understanding of core React concepts with room to improve on advanced topics.

## Key Strengths
- Strong grasp of React hooks fundamentals and purpose
- Clear communication style, explains concepts well

## Areas for Improvement
- closures: needs deeper understanding of lexical scope
- useEffect: dependency array behavior not fully internalized

## Final Recommendation
Hire — Candidate shows solid foundational knowledge. With some targeted study on closures and useEffect, they would be ready for mid-level React work.

## Interviewer Notes
Performance trend was improving across the session. Started cautious on easy questions but gained confidence mid-interview.
```

**Log output:**
```
Generating final report  { questionsAsked: 5, maxQuestions: 5, stopped: false }
LLM response received  { durationMs: 1240, totalTokens: 387 }
```

---

## 10. LLM Factory & Fallback Chain

**File:** `lib/llm.js`

All LLM calls in the interview agent go through `createLLMWithFallback()` which tries providers in order:

```
1. Groq (llama-3.3-70b-versatile)      — fastest (~600-900ms), free tier
2. OpenRouter (llama-3.3-70b)           — fallback if Groq rate-limits
3. Gemini (gemini-2.5-flash)            — final fallback
```

Used in `interviewAgent.js` via the factory:
```js
function makeLLM(temperature, maxTokens, streaming = false) {
    return createLLMWithFallback({
        provider: "groq",
        temperature,
        maxTokens,
        streaming,
        callbacks: [globalLLMLogger],   // timing + token logging
    });
}
```

### LLM Logger (`globalLLMLogger`)

Attached to every LLM instance via `callbacks: [globalLLMLogger]`. Fires on every LLM call:

```
LLM request started  { provider: "groq", model: "llama-3.3-70b-versatile" }
LLM response received  { durationMs: 850, promptTokens: 312, completionTokens: 68, totalTokens: 380 }
```

### LLM calls per turn summary

| Node | Temp | MaxTokens | Purpose |
|---|---|---|---|
| `planCurriculum` | 0.3 | 350 | Generate N curriculum slots as JSON |
| `processAnswer` | 0.1 | 300 | Intent detection + answer scoring as JSON |
| `handleEdgeCase` | 0.85 | 120 | Unexpected intent redirection (rare) |
| `generateQuestion` | 0.85 | 100 | Generate question text (streaming) |
| `generateFinalReport` | 0.3 | 400 | Markdown debrief |
| Profile update | 0.3 | 1000 | Extract insights for user memory |

---

## 11. Deepgram STT — How Speech Becomes Text

**File:** `agents/interview/stt.js`

### Connection
```js
wss://api.deepgram.com/v1/listen?
  encoding=linear16&
  sample_rate=48000&   // match LiveKit output
  channels=1&
  model=nova-3&
  smart_format=true&   // auto-punctuation, numbers as digits
  interim_results=true& // partial results while speaking
  endpointing=300       // 300ms silence = end of utterance
```

Configured via `.env`:
- `DEEPGRAM_STT_MODEL` (default: `"nova-3"`)
- `DEEPGRAM_STT_LANGUAGE` (default: `"en"`)
- `DEEPGRAM_STT_SMART_FORMAT` (default: `"true"`)
- `DEEPGRAM_STT_ENDPOINTING_MS` (default: `"300"`)

### Audio flow
```js
// worker.js:
for await (const frame of audioStream) {
    stt.pushAudio(frame.data);    // Int16Array: 48kHz PCM from LiveKit
}

// stt.js:
pushAudio(int16Array) {
    const buf = Buffer.from(int16Array.buffer, int16Array.byteOffset, int16Array.byteLength);
    this.socket.send(buf);        // binary over WebSocket to Deepgram
}
```

### Transcript assembly
Deepgram sends `Results` messages as the user speaks:
- `is_final=false`: interim result (discarded)
- `is_final=true`: final for this segment → `currentTranscript += text + " "`
- `speech_final=true`: user stopped speaking → `emit("transcript", currentTranscript.trim())` then reset

### Keepalive
Every 8 seconds, sends `{ type: "KeepAlive" }` to prevent Deepgram from closing the WebSocket during long AI responses (Deepgram's idle timeout is ~10-12s).

### Auto-reconnect
If the WebSocket closes unexpectedly and `_stopped=false`, reconnects after 1 second.

**Log output:**
```
STT connecting to Deepgram  { model: "nova-3" }
STT Deepgram WebSocket opened  { ms: 215, model: "nova-3" }
STT speech final  { transcript: "React hooks are functions that let you...", words: 12 }
```

---

## 12. Polly TTS — How Text Becomes Speech

**File:** `agents/interview/tts.js`

### `generatePCM(text, sampleRate="16000")`
Single AWS Polly call. Returns `Int16Array` of raw PCM audio.

```js
const command = new SynthesizeSpeechCommand({
    Text: text,
    OutputFormat: "pcm",          // raw uncompressed PCM (not MP3)
    SampleRate: "16000",          // 16kHz mono — what LiveKit AudioPublisher expects
    VoiceId: "Joanna",            // default, configurable via POLLY_VOICE_ID
    Engine: "neural",             // highest quality, configurable via POLLY_ENGINE
});
```

Configured via `.env`:
- `POLLY_VOICE_ID` (default: `"Joanna"`)
- `POLLY_ENGINE` (default: `"neural"`)
- `AWS_REGION` (default: `"us-east-1"`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

**Log output:**
```
TTS Polly request  { textLength: 47, text: "That's a good start, but you missed the...", voice: "Joanna", engine: "neural", region: "us-east-1" }
TTS Polly ready    { samples: 34560, durationMs: 380, voice: "Joanna", engine: "neural" }
```

### `generatePCMPipelined(text, sampleRate="16000")` — async generator
Splits text into sentences, fires ALL Polly requests concurrently, then yields PCM in sentence order.

```
Text: "Great answer. You correctly identified all three hooks. Now let's move on to a harder topic."
Sentences: ["Great answer.", "You correctly identified all three hooks.", "Now let's move on to a harder topic."]

T=0ms:  All 3 Polly requests start concurrently
T=330ms: Sentence 0 ready → yield { pcm, index: 0 }  [playback starts immediately]
T=360ms: Sentence 1 ready → yield { pcm, index: 1 }  [plays while sentence 0 finishes]
T=390ms: Sentence 2 ready → yield { pcm, index: 2 }
```

Caller (`worker.js`) receives each sentence as soon as it's ready and plays it:
```js
for await (const { pcm } of generatePCMPipelined(spokenText, "16000")) {
    await this._playAudio(pcm);   // blocks until each sentence finishes playing
}
```

**Log output:**
```
TTS pipeline start       { sentences: 3, totalChars: 92 }
TTS sentence dispatched  { idx: 0, sentence: "Great answer." }
TTS sentence dispatched  { idx: 1, sentence: "You correctly identified..." }
TTS sentence dispatched  { idx: 2, sentence: "Now let's move on..." }
TTS sentence ready       { idx: 0, durationMs: 330, voice: "Joanna" }
TTS sentence ready       { idx: 1, durationMs: 355, voice: "Joanna" }
TTS sentence ready       { idx: 2, durationMs: 388, voice: "Joanna" }
```

---

## 13. AudioPublisher — How PCM Reaches the Browser

**File:** `agents/shared/audioPublisher.js`

Wraps a `LiveKit AudioSource` and delivers PCM to it in precisely timed 10ms chunks.

```js
constructor(sampleRate = 16000, channels = 1) {
    this.samplesPerChunk = Math.floor(16000 * 10 / 1000);  // = 160 samples per 10ms
    this.source = new AudioSource(16000, 1);                // LiveKit source
}
```

### `pushPCM(pcmData: Int16Array)`

```js
for (let offset = 0; offset < pcmData.length; offset += 160) {
    const chunkData = pcmData.slice(offset, offset + 160);  // must use .slice() not .subarray()
    const frame = new AudioFrame(chunkData, 16000, 1, 160);
    await source.captureFrame(frame);

    // Precise pacing: calculate exact delay to next 10ms boundary
    const nextChunkTime = startTime + (chunksSent * 10);
    const delay = nextChunkTime - performance.now();
    if (delay > 0) await sleep(delay);
}
```

Key detail: Must use `.slice()` NOT `.subarray()`. The `livekit-rtc-node` library reads from `data.buffer` byte 0, so sharing an ArrayBuffer (via subarray) would cause every frame to contain the same first 160 samples — resulting in a looping glitch sound.

The `isSpeaking` flag is set to `true` during playback and `false` when done. The worker checks `audioPublisher.isSpeaking` before forwarding audio to Deepgram (to prevent echo/VAD issues).

**Log output:**
```
AudioPublisher playback start     { samples: 34560, sampleRate: 16000 }
AudioPublisher playback complete  { chunksSent: 216, playbackMs: 2162 }
```

---

## 14. TTS Phrase Cache System

Pre-recorded MP3 files in `assets/tts/` are used for common phrases to avoid Polly latency for short responses.

| Key | File | Spoken text |
|---|---|---|
| `great_answer` | `assets/tts/great_answer.mp3` | "Great answer!" |
| `good_effort` | `assets/tts/good_effort.mp3` | "Good effort." |
| `interesting` | `assets/tts/interesting.mp3` | "Interesting." |
| `take_your_time` | `assets/tts/take_your_time.mp3` | "Take your time." |
| `no_worries` | `assets/tts/no_worries.mp3` | "No worries." |
| `thats_okay` | `assets/tts/thats_okay.mp3` | "That's okay." |
| `lets_move_on` | `assets/tts/lets_move_on.mp3` | "Let's move on." |
| `next_question` | `assets/tts/next_question.mp3` | "Next question." |
| `final_question` | `assets/tts/final_question.mp3` | "This is the final question." |
| `interview_intro` | `assets/tts/interview_intro.mp3` | "Welcome to your interview." |
| `interview_outro` | `assets/tts/interview_outro.mp3` | "That concludes our interview." |
| `interview_stopped` | `assets/tts/interview_stopped.mp3` | "The interview has been stopped." |
| `thanks_for_time` | `assets/tts/thanks_for_time.mp3` | "Thanks for your time." |
| `out_of_context` | `assets/tts/out_of_context.mp3` | "Let's stay on topic." |

**How they're used in the browser path:**
The `worker.js` doesn't play MP3s directly over LiveKit (it needs PCM). Instead, it maps tags to `PHRASE_TEXT` strings and TTS's them inline via Polly:
```js
PHRASE_TEXT["good_effort"] = "Good effort."
// → this short text goes through generatePCM() → 150ms Polly call → plays via LiveKit
```

The MP3 files are used when sending `play_tts` socket events to the browser (for certain edge case intents like stop/unwell). The browser plays the MP3 via `<audio>`.

To regenerate the MP3 files: `node scripts/generateTTSCache.js`

---

## 15. Socket.io Events Reference

### Server → Browser

| Event | Payload | When emitted |
|---|---|---|
| `ai_state` | `{ state: "thinking\|speaking\|listening", text: "..." }` | State transitions during a turn |
| `transcript_final` | `{ role: "user\|ai", text: "..." }` | User transcript received; AI question ready |
| `ai_feedback` | `{ feedback, score, answerQuality, topicTag, followUpcoming }` | After processAnswer |
| `ai_stream` | `{ token: "...", type: "question\|follow_up\|edge_case" }` | During LLM streaming |
| `ai_stream_end` | `{ type: "question\|follow_up\|edge_case" }` | LLM stream complete |
| `play_tts` | `{ key: "...", audio: "<base64 mp3>" }` | Edge case with cached MP3 (stop/unwell) |
| `interview_done` | `{ report: "<markdown string>" }` | Interview completed |
| `voice_transcript` | `{ role: "ai\|user", text: "..." }` | Voice agent only (not interview) |
| `voice_state` | `{ state: "listening\|speaking" }` | Voice agent only |
| `voice_error` | `{ error: "..." }` | Voice agent only |

### Browser → Server

| Event | When |
|---|---|
| `register_session` (with `sessionId`) | When browser connects and wants interview updates |
| `client_audio_ready` (with `sessionId`) | Browser LiveKit room is connected and mic is active |

---

## 16. Full LangGraph State — All 24 Fields

Every field is a LangGraph channel — changes in nodes are merged with the last state value:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sessionId` | string | `""` | Ties to sessionCache, vectorStore, LiveKit room |
| `candidateName` | string | `"there"` | From `req.user.name` or "there" |
| `timeGreeting` | string | `"Hello"` | "Good morning/afternoon/evening" based on hour |
| `maxQuestions` | number | `5` | Target number of questions |
| `chatHistory` | Message[] | `[]` | LangChain HumanMessage/AIMessage objects |
| `conversationSummary` | string | `""` | "✓ topic: 8/10 \| ~..." — compact turn summary |
| `currentQuestion` | string | `""` | The last question asked by the AI |
| `userAnswer` | string | `""` | The transcribed user answer for current turn |
| `evaluation` | object | `null` | `{ score, accuracy, clarity, depth, feedback, nextDifficulty }` |
| `difficultyLevel` | string | `"medium"` | Current: "easy" \| "medium" \| "hard" |
| `questionsAsked` | number | `0` | How many planned questions have been asked |
| `topicsUsed` | string[] | `[]` | Topics covered (for "no repeat" rule) |
| `finalReport` | string \| null | `null` | Markdown debrief when done |
| `interviewStopped` | boolean | `false` | True if user said stop or is unwell |
| `intent` | string | `"normal"` | Last detected intent |
| `scores` | number[] | `[]` | All answer scores in sequence |
| `prefetchedContext` | string | `""` | Pre-loaded Qdrant context for next question |
| `repeatCount` | number | `0` | Consecutive nervous/irrelevant responses |
| `answerQuality` | string | `"normal"` | "correct" \| "partial" \| "wrong" \| "skipped" |
| `questionPlan` | object[] | `[]` | `[{ topic, difficulty, angle }, ...]` — curriculum |
| `topicTag` | string | `""` | Topic of current question (e.g. "React hooks") |
| `keyConceptMissed` | string | `""` | Specific concept missed (drives follow-up) |
| `topicScores` | object | `{}` | `{ "React hooks": [8, 6], "closures": [3] }` |
| `followUpAsked` | boolean | `false` | Whether a follow-up was asked for this slot |
| `followUpFlag` | boolean | `false` | If true, generateQuestion asks follow-up not next |
| `struggleStreak` | number | `0` | Consecutive wrong answers |
| `userProfileContext` | string | `""` | Injected at start from user_profiles table |

---

## 17. Phase 6 — Interview Completion & Final Report

When `generateFinalReportNode` runs:

1. LangGraph returns `{ finalReport: "<markdown>" }` — this is the last graph traversal
2. `sessionBridge.processUserTranscript()` returns `{ done: true, finalReport: "..." }`
3. Back in `handleUserTurn()`:
   ```js
   if (result.done) {
       io.to(sessionId).emit("interview_done", { report: result.finalReport });
       await session._onInterviewComplete(result);
       // Play outro TTS
       for await (const { pcm } of generatePCMPipelined(
           "Thank you for your time. The interview is now complete. You can review your report on the screen."
       )) {
           await this._playAudio(pcm);
       }
       setTimeout(() => this.stop(), 2000);
   }
   ```

4. `session._onInterviewComplete(result)` (defined in `routes/interview.js`):
   - `getDocumentBySessionId(sessionId)` → fetch document from Supabase
   - `saveInterviewResult({ userId, documentId, threadId, questionsAsked, scores, topicScores, finalReport, difficultyLevel })` → insert into Supabase `interview_results`
   - `logActivity({ action: "interview_completed", ... })` → insert into Supabase `activities`
   - `updateUserProfileAfterInterview(userId, finalState, docName)` → fire-and-forget profile update

5. `agent.stop()` after 2 seconds:
   - `stt.stop()` → close Deepgram WebSocket
   - `audioPublisher.stop()` → set stopFlag
   - `room.disconnect()` → leave LiveKit room

---

## 18. Phase 7 — User Profile Update (Cross-Session Memory)

**File:** `lib/interview/profileUpdater.js`

This runs after every completed interview (fire-and-forget, doesn't block the user).

### What it does:

**Step 1 — Fetch existing profile**
```js
const existing = await getUserProfile(userId);
// Supabase user_profiles table:
// { topics_covered, weak_areas, strong_areas, score_history, performance_summary, total_interviews, avg_score }
```

**Step 2 — LLM extracts insights from this interview**
```js
const prompt = `Analyze this interview result and extract structured insights.
Document: React Hooks Masterclass.pdf
Questions asked: 5
Topic scores: { "React hooks": [8,7], "useEffect": [6], "closures": [3,4] }
Final report: [first 1500 chars of markdown report]
Previous profile: [existing profile data if any]

Respond with ONLY valid JSON:
{
  "topics_covered": ["React hooks", "useEffect", "closures"],
  "weak_areas": ["closures: lexical scope not understood"],
  "strong_areas": ["React hooks: solid fundamental understanding"],
  "avg_score_this_session": 5.6,
  "performance_summary": "..."
}`;
```

**Step 3 — Merge with existing profile**
- Deduplicates `topics_covered`, `weak_areas`, `strong_areas` (case-insensitive)
- Cross-removes: if topic was previously weak but is now strong → removed from weak
- Running average: `newAvg = (prevAvg * prevTotal + thisScore) / (prevTotal + 1)`
- Score history: appends `{ date, score, topic, difficulty, questionsAsked }`, keeps last 50

**Step 4 — Upsert to Supabase**
```js
await upsertUserProfile(userId, {
    topics_covered: [...],
    weak_areas: [...],
    strong_areas: [...],
    score_history: [...],
    performance_summary: "...",
    total_interviews: 4,
    avg_score: 6.2,
    last_session_at: "2026-03-25T12:00:00Z"
});
```

**Next interview:** `getUserProfileContext(userId)` reads this profile and injects it as context, so the curriculum planner will:
- Ask MORE questions about `closures` (weak area)
- Ask LESS about `React hooks` (already strong)
- Adjust wording based on performance trend

---

## 19. Complete Flow Diagram

```
User opens browser
    │
    ├── Upload PDF ──→ PDF extracted → chunked → embedded → Qdrant + sessionCache
    │
    └── Click "Start Interview"
            │
            POST /api/interview/start
            │
            ├── ensureSession()                    load sessionCache or restore from DB
            ├── registerVectorStore()               module-level map for RAG access
            ├── getUserProfileContext()              load cross-session memory if auth'd
            ├── Build initialState                  sessionId, maxQ, difficulty, name, profile
            │
            ├─── PARALLEL ──────────────────────────────────────────────────────────┐
            │                                                                        │
            │  LangGraph.invoke(initialState)         agent.start()                 │
            │    routeOnStart → planCurriculum           [1/4] Deepgram STT start   │
            │    1 LLM call → questionPlan[]             [2/4] LiveKit token        │
            │    → generateQuestion (Q1)                 [3/4] LiveKit room connect │
            │    1 LLM call → first question text        [4/4] Publish audio track  │
            │    → END (resultState)                   setupRoomEvents()            │
            │                                                                        │
            └─────────────────────────────────────────────────────────────────────┘
                    │
            Wait for client_audio_ready (2s timeout)
                    │
            agent.speak("Hello! Welcome... " + Q1)
              parseTTSResponse → PHRASE_TEXT + uniquePart
              generatePCM(feedbackText) + generatePCM(questionText) in parallel
              audioPublisher.pushPCM() → 10ms chunks → LiveKit → browser speaker
                    │
            HTTP response { questionNumber: 1, difficulty: "medium", agentStarted: true }
            ┌─────────────────────────────────────────────────────────────────────────┐
            │                    REPEATING LOOP PER ANSWER                           │
            │                                                                        │
            │  Browser mic → LiveKit → AudioStream → stt.pushAudio()               │
            │      → Deepgram WS → speech_final → "transcript" event                │
            │                                                                        │
            │  handleUserTurn(transcript)                                            │
            │    🎤 User said log                                                   │
            │    emit ai_state:thinking                                              │
            │                                                                        │
            │    sessionBridge.processUserTranscript(transcript)                    │
            │      agentWorkflow.invoke({ userAnswer: transcript })                 │
            │                                                                        │
            │      routeOnStart → processAnswer                                      │
            │        [silence?] → return nervous instantly                           │
            │        [no]       → RAG getContext + LLM call                         │
            │                     intent / score / quality / nextDifficulty / topic │
            │                     followUpFlag? topicScores? struggleStreak?        │
            │                     emit ai_feedback to socket                        │
            │                                                                        │
            │      [non-normal intent] → handleEdgeCase                             │
            │        cached phrase OR 1 LLM call                                    │
            │        [stopped/last] → generateFinalReport → END                     │
            │        [continue]     → generateQuestion → END                        │
            │                                                                        │
            │      [normal intent] ──→ PARALLEL:                                    │
            │        prefetchContext (RAG for next topic)                            │
            │        adaptDifficulty (easy/medium/hard logic)                       │
            │        → updateSummary (build "✓ topic:8" string, trim history)       │
            │        → [questionsAsked >= maxQ?] generateFinalReport               │
            │        → [else]                   generateQuestion                    │
            │             [followUpFlag?] → probe missed concept                   │
            │             [else]          → next curriculum slot                   │
            │             LLM call (streaming) → tokens → socket.emit ai_stream    │
            │             → END                                                     │
            │                                                                        │
            │    result = { done, evaluation, nextQuestion, intent, ... }           │
            │                                                                        │
            │    [done?] → emit interview_done → save DB → play outro → stop()     │
            │    [no]    → parse TTS tags → log cache hits                          │
            │              parallel TTS (feedback + question)                       │
            │              audioPublisher.pushPCM() → LiveKit → browser speaker    │
            │              emit ai_state:listening                                  │
            │                                                                        │
            └─────────────────────────────────────────────────────────────────────┘
                    │
            Interview done
                    │
            generateFinalReportNode
              pre-aggregate: avg, best, worst, trend, grade, per-topic
              1 LLM call → markdown debrief
                    │
            session._onInterviewComplete()
              saveInterviewResult() → Supabase
              logActivity()
              updateUserProfileAfterInterview() [fire-and-forget]
                1 LLM call → extract insights
                merge weak/strong areas
                upsert user_profiles
                    │
            agent.stop()
              stt.stop() → close Deepgram WS
              audioPublisher.stop()
              room.disconnect() → leave LiveKit
```

---

## 20. Timing & Performance

### Startup latency (from "Start Interview" click to first spoken word)

| Step | Typical time |
|---|---|
| `planCurriculum` LLM call (Groq) | ~700-1000ms |
| `generateQuestion` LLM call (Groq) | ~600-800ms |
| LiveKit room connect (parallel) | ~150-300ms |
| Polly TTS for first question | ~400-600ms |
| Client audio ready signal | ~500-1000ms |
| **Total** | **~2.5-4s** |

The LangGraph first invocation and LiveKit startup overlap, saving ~300-400ms.

### Per-turn latency (user finishes speaking → AI starts speaking)

| Step | Typical time |
|---|---|
| Deepgram endpointing (silence detection) | 300ms (configurable) |
| Deepgram transcription | ~50-100ms |
| `processAnswer` LLM call | ~700-900ms |
| `generateQuestion` LLM call | ~600-800ms |
| Polly TTS (first sentence via pipeline) | ~350-500ms |
| **Total before first word spoken** | **~2.0-2.6s** |

The parallel prefetchContext + adaptDifficulty during `processAnswer` saves ~150ms per turn (Qdrant is pre-warmed).

### Key performance optimizations

1. **Parallel startup:** `Promise.all([interviewAgent.invoke, agent.start()])` — saves ~400ms
2. **Pipelined TTS:** All sentences sent to Polly concurrently — saves ~300ms per response
3. **Parallel feedback+question TTS:** Feedback plays while question TTS generates — saves ~350ms
4. **Context prefetching:** Qdrant fetched during LLM evaluation — saves ~150ms per turn
5. **RAG cache:** `ragCache` Map caches identical queries — eliminates repeated Qdrant calls
6. **Deepgram keepalive:** Prevents reconnection delays during AI speech
7. **PostgresSaver:** LangGraph state persisted after every node — allows server restart recovery
