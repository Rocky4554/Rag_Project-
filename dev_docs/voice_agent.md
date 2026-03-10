# Voice AI Interview Agent — Implementation Plan

## What We Are Building

A full **AI Voice Interview Agent** that sits on top of the existing RAG + Quiz app.

**User Flow:**
1. User uploads a PDF (already works)
2. User starts an "Interview Mode"
3. AI generates a question from the PDF → speaks it aloud (Kokoro TTS)
4. User speaks their answer into the microphone (Web Speech API)
5. AI evaluates the answer → gives structured feedback score → speaks feedback aloud
6. AI adapts the next question based on how well the user answered
7. After N questions → AI generates a Final Report with overall grade and breakdown

---

## Architecture Diagram

```
PDF
  ↓
ChromaDB (already exists)
  ↓
[POST /api/interview/start]
  ↓
LangGraph Interview Agent
  ↓ ↓ ↓ ↓ ↓ ↓ ↓
  Node 1: generateQuestion    → picks topic from retrieved PDF chunks
  Node 2: questionToSpeech    → Kokoro TTS → sends MP3 to browser
  Node 3: waitForUserAnswer   → frontend sends recorded answer text
  Node 4: evaluateAnswer      → Gemini scores: accuracy, clarity, depth
  Node 5: feedbackToSpeech    → Kokoro TTS → sends feedback MP3
  Node 6: adaptNextQuestion   → harder if good, simpler if struggling
  Node 7: finalReport         → after N rounds, generate full report
```

---

## Tech Stack (What Is New vs What Already Exists)

### Already Exists ✅
| What | Package |
|---|---|
| PDF upload & parsing | `pdf-parse` |
| Text splitting | `@langchain/textsplitters` |
| Vector DB | `chromadb` + `@langchain/community` |
| Gemini LLM | `@langchain/google-genai` |
| TTS (speech output) | Kokoro-FastAPI Docker (already running) |
| Express server | `express` |

### New Things to Add 🆕
| What | How |
|---|---|
| **LangGraph** (interview orchestration) | `@langchain/langgraph` npm package |
| **Speech-to-Text (STT)** | Browser native `Web Speech API` — zero cost, zero install |
| **Interview state management** | In `sessionCache[sessionId].interviewState` |
| **Structured scoring** | Gemini JSON output (same pattern as quizGenerator.js) |
| **Adaptive difficulty** | State variable `difficultyLevel` in LangGraph state |

---

## New npm Dependency

```bash
npm install @langchain/langgraph
```

That is the **only** new npm package. Everything else either already exists or is free browser APIs.

---

## File Plan

### 1. [NEW] `lib/interviewAgent.js`

This is the heart of the feature. It creates and runs a **LangGraph state machine**.

**LangGraph State Schema:**
```js
{
  sessionId: string,
  vectorStore: object,       // from sessionCache
  chatHistory: [],           // LangChain message history
  currentQuestion: string,   // current question text
  userAnswer: string,        // latest answer from user
  evaluation: {              // structured scoring
    score: number,           // 1-10
    accuracy: number,        // 1-10
    clarity: number,         // 1-10
    depth: number,           // 1-10
    feedback: string,        // textual feedback
    nextDifficulty: string   // "easier" | "same" | "harder"
  },
  difficultyLevel: string,   // "easy" | "medium" | "hard"
  questionsAsked: number,    // counter
  maxQuestions: number,      // default 5
  topicsUsed: [],            // to avoid repeating same topic
  finalReport: null          // set at end
}
```

**LangGraph Nodes:**

```js
// Node 1 — generateQuestion
// Retrieves 5 chunks from ChromaDB based on difficultyLevel
// Asks Gemini to write one question at that difficulty
// Avoids repeating topicsUsed

// Node 2 — evaluateAnswer
// Receives userAnswer, currentQuestion, context
// Asks Gemini to return JSON with score, accuracy, clarity, depth, feedback, nextDifficulty

// Node 3 — adaptNextQuestion
// Edge logic: if questionsAsked >= maxQuestions → route to finalReport
//             else → route to generateQuestion with updated difficulty

// Node 4 — generateFinalReport
// Collects all Q&A pairs and scores from chatHistory
// Asks Gemini to produce a formal interview report
```

**LangGraph Graph structure:**

```js
const graph = new StateGraph(InterviewState)
  .addNode("generateQuestion", generateQuestion)
  .addNode("evaluateAnswer", evaluateAnswer)
  .addNode("generateFinalReport", generateFinalReport)
  .addEdge(START, "generateQuestion")
  .addEdge("generateQuestion", END)   // pauses, waits for user
  .addEdge("evaluateAnswer", "generateQuestion")  // or finalReport
  .addConditionalEdges("evaluateAnswer", adaptNextQuestion)
  .compile();
```

> **Key insight:** LangGraph will **pause** after `generateQuestion` and return to the frontend. The frontend sends the user's answer back via `/api/interview/answer`, which resumes the graph at `evaluateAnswer`.

---

### 2. [NEW] `lib/speechToAudio.js`

Simple wrapper to convert any text → Kokoro TTS → base64 MP3. Extracted from `summarizer.js` to be shared between the summary feature and the interview agent.

**Exports:** `textToAudio(text)` (same logic as in summarizer.js — refactored to be reusable)

---

### 3. [MODIFY] `server.js`

Two new API routes:

**`POST /api/interview/start`**
```
Body: { sessionId, maxQuestions: 5 }
→ Initializes the LangGraph state for this session
→ Runs the graph up to generateQuestion (first question)
→ Calls Kokoro TTS on the question
→ Returns: { questionText, questionAudio (base64), questionNumber: 1 }
```

**`POST /api/interview/answer`**
```
Body: { sessionId, answerText }
→ Resumes the LangGraph with the user's answer
→ Runs evaluateAnswer → adaptNextQuestion → generateQuestion (or finalReport)
→ Calls Kokoro TTS on the feedback text
→ Returns: { evaluation, feedbackAudio (base64), nextQuestion, nextQuestionAudio, done: false }
OR:         { finalReport, done: true }
```

Update `sessionCache` structure:
```js
sessionCache[sessionId] = {
  vectorStore,
  docs,
  chatHistory: [],          // already there
  interviewState: null,     // NEW — stores LangGraph state between requests
};
```

---

### 4. [MODIFY] `public/index.html`

Add a new **Interview Mode** card that appears after upload, alongside the existing Summary and Quiz sections.

**UI Flow:**
1. "Start Interview" button → calls `/api/interview/start`
2. Audio plays the first question (using `<audio>` tag)
3. "Record Answer" button appears → activates Web Speech API `SpeechRecognition`
4. Transcript appears in a text box as user speaks
5. "Submit Answer" button → calls `/api/interview/answer` with transcript
6. Feedback audio plays, next question appears
7. After final question → Final Report card appears with scores and summary

**Key Web Speech API code:**
```js
const recognition = new webkitSpeechRecognition() || new SpeechRecognition();
recognition.continuous = false;
recognition.interimResults = true;
recognition.lang = 'en-US';
recognition.onresult = (event) => {
  transcript = event.results[event.results.length - 1][0].transcript;
  document.getElementById('answer-text').value = transcript;
};
```

> Chrome and Edge support the Web Speech API natively with no API key and no cost.

---

## Adaptive Difficulty Prompt (Key Prompts to Use)

### Question Generation Prompt
```
You are interviewing a candidate based on the following PDF content.
Difficulty level: {difficultyLevel}

easy = factual recall, basic definitions
medium = explanation and application 
hard = analysis, comparison, edge cases

Previously asked topics: {topicsUsed}
Do NOT repeat those topics.

Context: {context}

Generate exactly ONE {difficultyLevel} interview question. Return only the question text.
```

### Answer Evaluation Prompt
```
You are an expert interviewer evaluating a candidate's answer.

Question: {question}
Candidate's Answer: {answer}
Reference Context: {context}

Evaluate the answer and return ONLY a valid JSON object:
{
  "score": <1-10>,
  "accuracy": <1-10>,
  "clarity": <1-10>,
  "depth": <1-10>,
  "feedback": "<2-3 sentence constructive feedback>",
  "nextDifficulty": "<easier|same|harder>"
}
```

### Final Report Prompt
```
You are summarizing an AI interview session. Here is the full Q&A history:
{qa_history}

Generate a professional interview report with:
- Overall Score (average of all scores)
- Strengths shown by the candidate
- Areas for Improvement
- Final Recommendation: Pass / Borderline / Needs Improvement
Return the report in plain text paragraphs (no markdown).
```

---

## State Between Requests

Because HTTP is stateless, the LangGraph interview state is saved in `sessionCache`:

```js
sessionCache[sessionId].interviewState = {
  graphState,        // full LangGraph state snapshot
  questionNumber,    // which question we're on
  scores: [],        // all evaluation objects collected
}
```

This persists in memory as long as the server is running (same as the existing vectorStore and docs).

---

## Implementation Order (Step by Step)

1. `npm install @langchain/langgraph`
2. Create `lib/speechToAudio.js` (refactor TTS out of summarizer.js)
3. Create `lib/interviewAgent.js` (LangGraph state machine)
4. Add two routes to `server.js`
5. Add Interview UI to `public/index.html`
6. Test locally end-to-end

---

## What You Will Be Able to Demonstrate After This

| Feature | Tech Used |
|---|---|
| Retrieval-Augmented Generation | LangChain + ChromaDB |
| Multi-step AI agent with state | LangGraph |
| Speech output | Kokoro TTS |
| Speech input | Web Speech API |
| Adaptive AI logic | LangGraph conditional edges |
| Structured AI output | Gemini JSON mode |
| Real-time interview loop | Express REST + in-memory state |

---

## Estimated Effort

| Task | Effort |
|---|---|
| `lib/interviewAgent.js` | 3–4 hours |
| `server.js` routes | 1 hour |
| `public/index.html` Interview UI | 2 hours |
| End-to-end testing | 1 hour |
| **Total** | **~7 hours** |
