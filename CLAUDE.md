# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-powered AI learning + voice interview platform. Users upload a PDF, image, or pasted text; the system builds a Qdrant vector store + BM25 index, then offers chat, summary (with audio), quizzes, and an adaptive voice interview over LiveKit driven by a LangGraph state machine. The Next.js UI lives in the sibling `rag_frontend/` repo; `public/index.html` is an older single-file UI.

## Commands

```bash
node server.js          # Express + Socket.io (PORT, default 3000)
npm run dev             # same, with nodemon
npm run tts:cache       # regenerate cached TTS phrase MP3s (assets/tts/)
```

Voice agents are not separate processes — routes start them in-process (`agents/*/worker.js`). There are no build, lint, or test commands.

## Project Structure

```
server.js                  # Express setup, Socket.io, rate limiters, route mounting, session TTL cleanup
routes/                    # Each exports a factory receiving shared deps → Express Router
  auth.js                  # signup, login, refresh, logout, profile, forgot-password / verify-otp / reset-password
  upload.js                # POST /upload (PDF or image), POST /upload/text
  chat.js                  # POST /chat, POST /chat/stream (SSE), GET /chat/history/:sessionId
  quiz.js, summary.js
  interview.js             # POST /interview/start, POST /interview/stop
  conversationalAI.js      # POST /conversational-ai/start|stop (Gemini Live voice agent)
  tokens.js                # POST /livekit/token
  history.js               # documents (list/delete), interviews, quizzes, activity
  session.js               # session restore / status / user-sessions
middleware/auth.js         # requireAuth / optionalAuth (JWT cookie)
lib/
  access.js                # canAccessSession (ownership), publicError (hide internals in prod)
  jwt.js                   # access/refresh tokens with a `type` claim
  passwordVersion.js       # `pwv` token claim — password reset invalidates old tokens
  llm.js                   # createLLMWithFallback: Gemini or Groq primary → OpenRouter Llama fallback
  embeddings.js            # Gemini text embeddings + Jina CLIP image embedder
  sessionRestore.js        # ensureSession: in-memory cache → restore from Supabase + Qdrant + BM25 file
  db.js, supabase.js, email.js, env.js, logger.js, validation.js (zod 4 — use `.issues`)
  pipeline/                # pdfLoader, textSplitter, vectorStore, bm25Index, hybridRetrieval, reranker,
                           # rag, queryRouter, imageChat
  interview/               # interviewAgent (classic LangGraph), deepInterviewAgent, quizGenerator,
                           # summarizer, greetingGenerator, profileUpdater
  tts/                     # provider registry; app code imports lib/tts/speechToAudio.js
agents/
  shared/voicePipelineWorker.js  # base: LiveKit room + Deepgram STT + TTS, turn queue, barge-in
  shared/audioPublisher.js
  interview/               # InterviewAgentWorker, sessionBridge (→ LangGraph), stt, tts, transcriptCleaner
  voice/worker.js          # VoiceAgentWorker (Gemini Live, conversational AI)
supabase/                  # SQL migrations
```

## Architecture

### Ingestion
PDF → pdf-parse (per page) → RecursiveCharacterTextSplitter (1500 chars, 200 overlap) → drop low-value + duplicate chunks → Gemini embeddings (batched, retried, SHA256 chunk cache in `.cache/embedding-cache.json`, LRU-capped by `EMBEDDING_CACHE_MAX_ENTRIES`) → Qdrant collection named after the sessionId. Chunks that fail to embed are dropped, never stored as zero vectors. A BM25 (minisearch) index is kept in memory and persisted to `.cache/bm25/`.

### Chat
`queryRouter` classifies DIRECT vs RAG (one-word LLM reply) **concurrently** with hybrid retrieval; DIRECT streams a general-knowledge answer, RAG answers from the retrieved docs. Hybrid retrieval (`hybridRetrieval.js`): BM25 + semantic (50 each) → Reciprocal Rank Fusion → Jina reranker → top 10. Without a BM25 index (e.g. restored session, missing file) it is semantic + rerank only.

### Interview (LangGraph, `lib/interview/interviewAgent.js`)
Nodes: planCurriculum, routerNode (intent), gradeAnswer, backchannel, handleEdgeCase, prefetchContext + adaptDifficulty (parallel), updateSummary, generateQuestion, generateFinalReport. `INTERVIEW_AGENT_MODE=deep` swaps in `deepInterviewAgent.js` (LLM orchestrator). Both use `getContext()` → the same hybrid retrieval as chat. Router regex fast-paths only match when the whole utterance is the phrase; everything else goes to the LLM classifier. A grade that can't be parsed after one retry is marked `ungraded` and excluded from scores. Checkpoints: PostgresSaver (Supabase), falling back to MemorySaver if Postgres is unavailable.

### Voice path
Browser mic → LiveKit → `VoicePipelineWorker` → Deepgram STT → `InterviewAgentWorker.onUserTranscript` → `SessionBridge` → LangGraph → TTS (Polly / Deepgram / ElevenLabs via `INTERVIEW_TTS_PROVIDER`) → `AudioPublisher` → LiveKit. Turns are processed one at a time (`_enqueueTurn`); a barge-in cancels the rest of the current turn's speech via an AsyncLocalStorage turn context. Fragments that look cut off are held and merged with the next utterance (or processed after 3s). `POST /interview/stop` ends via a system "stop" turn so the final report is still produced.

### Auth & access
Custom auth (bcrypt + JWT in httpOnly cookies; `JWT_SECRET` required). `optionalAuth` returns 401 for a present-but-invalid cookie so clients refresh. A session uploaded while logged in has `ownerId` and is only usable by that user (`canAccessSession`) — enforced on all session routes, the LiveKit token, and the Socket.io `register_session` (which emits `register_error` when refused). Anonymous uploads are reachable by sessionId alone.

### State
In-memory `sessionCache`, `activeAgents`, interview registries — the server is single-instance by design. Persistent: Supabase (users, documents, chat_messages, interview_results, quiz_results, activities, active_interviews, password_resets, user_profiles), Qdrant (vectors), LangSmith (traces).

### Socket.io events
Client → server: `register_session`, `client_audio_ready`. Server → client: `register_error`, `ai_state`, `transcript_final`, `ai_speech`, `ai_subtitle`, `ai_feedback`, `interview_done`, `voice_error`.

## Environment Variables

Required (startup fails without them): `GEMINI_API_KEY`, `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `JWT_SECRET`.
Feature groups: Supabase (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_URL`), LiveKit (`LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`), `DEEPGRAM_API_KEY`, AWS Polly credentials, `ELEVENLABS_API_KEY`, `JINA_API_KEY` (reranker + image uploads), `OPENROUTER_API_KEY` (LLM fallback), LangSmith (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`).
Tuning: `GEMINI_MINI_MODEL`, `INTERVIEW_AGENT_MODE`, `INTERVIEW_TTS_PROVIDER`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_CACHE_MAX_ENTRIES`, `SESSION_TTL_MS`, `CORS_ORIGINS`, `DEEPGRAM_STT_*`, `VAD_INSTANT_BARGE_IN`.
