# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-powered AI Voice Interview Platform. Users upload PDFs, the system builds a vector store, then conducts adaptive voice interviews over WebRTC using a LangGraph state machine.

## Commands

```bash
# Start the server (Express + Socket.io on PORT=3000)
node server.js

# Run CLI entry point
node index.js

# Start LiveKit voice agent worker
node agent/agent.js
```

There are no build steps, linting, or test commands configured.

## Architecture

### Core Pipeline

```
PDF Upload → text extraction (pdf-parse) → chunking (900 chars, 150 overlap)
  → deduplicate → embed (Gemini gemini-embedding-001, batched/cached)
  → store in Qdrant (per-session collections)
```

### LLM Fallback Chain

Groq (llama-3.3-70b) → OpenRouter (llama-3.3-70b) → Gemini (gemini-2.5-flash). Factory in `lib/llm.js`.

### LangGraph Interview State Machine (`lib/interviewAgent.js`)

8-node graph: `planCurriculum → processAnswer → handleEdgeCase → prefetchContext + adaptDifficulty (parallel) → updateSummary → generateQuestion → generateFinalReport`. State has ~20 channels tracking curriculum, scores, difficulty, follow-ups, and struggle streaks.

### Real-Time Voice Path

```
Browser mic → LiveKit WebRTC → agent/agent.js (InterviewAgentWorker)
  → agent/stt.js (Deepgram nova-3 via WebSocket) → transcript
  → agent/sessionBridge.js → LangGraph state machine
  → LLM generates response → agent/tts.js (Polly PCM)
  → agent/audioPublisher.js (10ms PCM chunks) → LiveKit → browser audio
```

### Key Modules

- **`server.js`** — Express REST API + Socket.io events. Holds in-memory `sessionCache` mapping sessionId → vectorStore, docs, chatHistory, interviewStateConfig.
- **`lib/rag.js`** — RAG query: prepends last 3 chat pairs, retrieves top 3 chunks from Qdrant, invokes LLM.
- **`lib/vectorStore.js`** — Qdrant integration with SHA256-based embedding cache (`.cache/embedding-cache.json`). Batches of 20, 250ms inter-batch delay, 3 retries with backoff.
- **`lib/tts/index.js`** — Provider registry. `TTS_PROVIDER` env var selects polly or kokoro. All app code imports from `lib/speechToAudio.js`.
- **`agent/`** — LiveKit worker components. `agent.js` is the main worker; `sessionBridge.js` bridges real-time audio events to the LangGraph graph.

### API Endpoints

- `POST /api/upload` — Upload & process PDF, returns sessionId
- `POST /api/chat` — RAG query with chat history
- `POST /api/quiz` — Generate quiz from document
- `POST /api/summary` — Summarize document + generate audio
- `POST /api/interview/start` — Initialize LangGraph interview
- `POST /api/livekit/token` — WebRTC access token
- `GET /api/deepgram/token` — STT config for client

### Socket.io Events

- `register_session` — Client subscribes to session updates
- `transcript_final` / `ai_feedback` / `interview_done` — Server pushes interview state to UI

### Frontend (`public/index.html`)

Single-file app with tabbed UI: Upload, Config, Summary, Chat, Interview (setup/active/report), Quiz, Results. Uses Socket.io client and LiveKit browser SDK (`livekit-client.umd.min.js`).

## Environment Variables

See `.env.example`. Required keys: `GEMINI_API_KEY`, `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`, `DEEPGRAM_API_KEY`, and AWS credentials for Polly TTS.
