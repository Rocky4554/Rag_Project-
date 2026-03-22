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

## Project Structure

```
server.js                         # Express app setup, Socket.io, route mounting
index.js                          # CLI entry point

routes/
  auth.js                         # POST /api/auth/signup, login, refresh, GET profile
  upload.js                       # POST /api/upload
  chat.js                         # POST /api/chat, GET /api/chat/history/:sessionId
  quiz.js                         # POST /api/quiz
  summary.js                      # POST /api/summary
  interview.js                    # POST /api/interview/start
  tokens.js                       # POST /api/livekit/token, GET /api/deepgram/token
  history.js                      # GET /api/history/documents, interviews, quizzes, activity

middleware/
  auth.js                         # requireAuth + optionalAuth (Supabase JWT)

lib/
  supabase.js                     # Supabase client (admin + per-user factory)
  db.js                           # DB helpers (documents, chat, interviews, quizzes, activity)
  llm.js                          # LLM factory with fallback chain (cross-cutting)
  embeddings.js                   # Gemini embedding model (cross-cutting)
  pipeline/                       # Document processing chain
    pdfLoader.js                  # PDF text extraction
    textSplitter.js               # Text chunking
    vectorStore.js                # Qdrant + embedding cache
    rag.js                        # RAG query logic
  interview/                      # Interview engine
    interviewAgent.js             # LangGraph state machine (8 nodes)
    quizGenerator.js              # Quiz generation from vector store
    summarizer.js                 # Document summarization
  tts/                            # Text-to-speech
    index.js                      # Provider registry
    speechToAudio.js              # Re-export (entry point for app code)
    providers/
      pollyProvider.js
      kokoroProvider.js

agent/                            # LiveKit voice agent worker
  agent.js                        # InterviewAgentWorker (main worker)
  audioPublisher.js               # Pushes PCM frames to LiveKit
  sessionBridge.js                # Bridges audio events to LangGraph
  stt.js                          # Deepgram STT via WebSocket
  tts.js                          # AWS Polly TTS (PCM for LiveKit)

scripts/                          # Utility & test scripts
public/                           # Frontend (single-file app)
assets/tts/                       # Cached TTS phrase MP3s
dev_docs/                         # Developer documentation
supabase/
  schema.sql                      # SQL migration for all Supabase tables + RLS
```

## Architecture

### Core Pipeline

```
PDF Upload → text extraction (pdf-parse) → chunking (900 chars, 150 overlap)
  → deduplicate → embed (Gemini gemini-embedding-001, batched/cached)
  → store in Qdrant (per-session collections)
```

### LLM Fallback Chain

Groq (llama-3.3-70b) → OpenRouter (llama-3.3-70b) → Gemini (gemini-2.5-flash). Factory in `lib/llm.js`.

### LangGraph Interview State Machine (`lib/interview/interviewAgent.js`)

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

- **`server.js`** — Express app setup, Socket.io events, mounts route files. Async `startServer()` initializes PostgresSaver checkpointer before mounting routes. Holds in-memory `sessionCache` (vectorStore, docs) with persistent data in Supabase.
- **`routes/`** — Each file exports a factory function receiving shared deps (sessionCache, io, etc.) and returning an Express Router.
- **`lib/pipeline/rag.js`** — RAG query: prepends last 3 chat pairs, retrieves top 3 chunks from Qdrant, invokes LLM.
- **`lib/pipeline/vectorStore.js`** — Qdrant integration with SHA256-based embedding cache (`.cache/embedding-cache.json`). Batches of 20, 250ms inter-batch delay, 3 retries with backoff.
- **`lib/tts/index.js`** — Provider registry. `TTS_PROVIDER` env var selects polly or kokoro. App code imports from `lib/tts/speechToAudio.js`.
- **`agent/`** — LiveKit worker components. `agent.js` is the main worker; `sessionBridge.js` bridges real-time audio events to the LangGraph graph.

### Persistence Layer

- **Supabase PostgreSQL** — users (auth), documents, chat_messages, interview_results, quiz_results, activities
- **LangGraph PostgresSaver** — interview agent checkpoints (auto-managed tables in same Supabase DB)
- **Qdrant** — vector embeddings only
- **LangSmith** — traces all LangGraph invocations with thread_id for observability

### API Endpoints

**Auth:**
- `POST /api/auth/signup` — Create account (email + password)
- `POST /api/auth/login` — Login, returns JWT session
- `POST /api/auth/refresh` — Refresh JWT token
- `GET /api/auth/profile` — User profile + documents + recent activity (auth required)

**Core:**
- `POST /api/upload` — Upload & process PDF, returns sessionId
- `POST /api/chat` — RAG query with chat history
- `GET /api/chat/history/:sessionId` — Retrieve persisted chat history (auth required)
- `POST /api/quiz` — Generate quiz from document
- `POST /api/summary` — Summarize document + generate audio
- `POST /api/interview/start` — Initialize LangGraph interview
- `POST /api/livekit/token` — WebRTC access token
- `GET /api/deepgram/token` — STT config for client

**History (auth required):**
- `GET /api/history/documents` — All user's uploaded documents
- `GET /api/history/interviews` — Interview results (optional ?documentId filter)
- `GET /api/history/quizzes` — Quiz results (optional ?documentId filter)
- `GET /api/history/activity` — User activity log

### Socket.io Events

- `register_session` — Client subscribes to session updates
- `transcript_final` / `ai_feedback` / `interview_done` — Server pushes interview state to UI

### Frontend (`public/index.html`)

Single-file app with tabbed UI: Upload, Config, Summary, Chat, Interview (setup/active/report), Quiz, Results. Uses Socket.io client and LiveKit browser SDK (`livekit-client.umd.min.js`).

## Environment Variables

See `.env.example`. Required keys: `GEMINI_API_KEY`, `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`, `DEEPGRAM_API_KEY`, AWS credentials for Polly TTS, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_URL`. Optional: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` for LangSmith.
