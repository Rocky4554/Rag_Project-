# Performance Optimizations

Findings from a full-codebase latency/perf audit (three parallel sweeps: voice
pipeline, RAG/upload pipeline, server/routes/frontend), with fixes applied for
the highest-impact, lowest-risk items. Each was read from source and verified
before changing; estimates below are from the audit, not measured in prod.

## Voice pipeline (time-to-first-audio, per-turn latency)

### Speech marks blocked the first audio chunk
`agents/shared/voicePipelineWorker.js` — `_speakAndEmit()`

For Polly, `getSpeechMarks(text)` is a **second full API round trip**
(`SynthesizeSpeechCommand` with `OutputFormat: "json"`), separate from the
audio synthesis call. The code awaited it before playing the first PCM
chunk:
```js
if (firstChunk) {
    firstChunk = false;
    const marks = await marksPromise;   // blocks first audio on a 2nd Polly call
    ...
}
```
Subtitles are cosmetic. Changed to emit them whenever they land via
`marksPromise.then(...)`, guarded by the speech epoch so marks for an
already-interrupted utterance are dropped instead of emitted late. Saves
~200-500ms of dead air per utterance.

### Fixed 120ms sleep on every barge-in
`agents/shared/voicePipelineWorker.js` — barge-in handler

```js
this._interruptSpeech();
await new Promise(r => setTimeout(r, 120));   // removed
```
`_interruptSpeech()` bumps `_speechEpoch` and calls `audioPublisher.stop()`,
which is synchronous. `_playAudio()` already checks
`epoch !== this._speechEpoch` and bails — so the in-flight playback loop
cannot emit stale audio past the interrupt point. The sleep had nothing left
to wait for. Removed; verified `audioPublisher.stop()`'s synchronicity by
reading its implementation before removing the guard.

### Gemini client rebuilt on every short-answer check
`agents/interview/transcriptCleaner.js` — `checkSemanticCompleteness()`

`new GoogleGenerativeAI(apiKey)` + `.getGenerativeModel(...)` were
constructed fresh on every call. This function runs in the voice hot path
for any transcript under 8 words with no terminal punctuation — a very
common shape for interview answers ("yes", "a hash map", "linked lists").
Memoized as a module-level singleton (`getCompletenessModel()`).

### Leaked timers in LLM timeout races
`lib/interview/interviewAgent.js` (`generateBackchannel`, 800ms race) and
`lib/interview/deepInterviewAgent.js` (orchestrator dispatch, 1800ms race)

Both used `Promise.race([llmCall, new Promise(r => setTimeout(r, N))])`
without clearing the timer when the LLM call won. `lib/embeddings.js` already
had the correct pattern (`.finally(() => clearTimeout(timer))`) — applied the
same fix to both. Low-impact but easy: an uncleared timer keeps the event
loop referenced for the timeout duration on every turn even when the LLM
responds instantly.

## Upload / RAG pipeline

### Embedding batches: serial + a sleep after every batch, including the last
`lib/pipeline/vectorStore.js`

`BATCH_CONCURRENCY` defaulted to `1` (fully serial batches of 20 chunks), and
a fixed 250ms `sleep` ran after *every* batch — including the final one,
where it's pure dead time on the upload request. Changed default concurrency
to `3` (the retry/backoff path in `embedBatchWithRetry` already isolates
429s per item, so this degrades gracefully) and skip the sleep after the
last batch: `if (batchIdx < batches.length - 1) await sleep(...)`.

### PDF written to /tmp and read back for no reason
`routes/upload.js`, `lib/pipeline/pdfLoader.js`

The uploaded file is already fully in memory (`multer` memory storage), but
the code wrote it to `os.tmpdir()`, passed the path to `PDFLoader`, then
unlinked it — a full disk write + read of the entire PDF on every upload.
`PDFLoader` accepts a `Blob` directly. `extractTextFromPDF()` now accepts
either a `Buffer` or a path (wraps the buffer in
`new Blob([buf], { type: 'application/pdf' })`); `routes/upload.js` passes
`req.file.buffer` straight through. Verified output is byte-identical
between the buffer path and the old file-path path on `sample.pdf`.

### BM25 index serialized three times over
`lib/pipeline/bm25Index.js` — `persistBM25Index()`

```js
index: JSON.parse(JSON.stringify(bm25Index.miniSearch)),  // calls toJSON()
```
then the whole `payload` object (including that already-stringified-and-
reparsed index) was `JSON.stringify`'d again for the write. `JSON.stringify`
already invokes `.toJSON()` on any object that defines it — MiniSearch does —
so the `JSON.parse(JSON.stringify(...))` wrapper was pure waste, and all of
it runs **synchronously**, blocking the event loop (and every other in-flight
request/socket) for the duration. Verified `JSON.stringify({index: ms})` and
`JSON.stringify({index: JSON.parse(JSON.stringify(ms))})` produce byte-
identical output before removing the wrapper.

## Server / routes

### Session sweeper crash (see `voice_agent_not_starting_fix.md`)
Not originally a perf item, but worth noting here too: an unbounded
`sessionCache` (each entry holding a `vectorStore`, BM25 index, and full
document text) was the direct consequence of the sweeper's `ReferenceError`
aborting every eviction cycle. Fixing the crash *is* the fix for this leak.

### Sequential DB deletes with no data dependency
`lib/db.js` — `deleteDocument()`

Three deletes against unrelated tables (`chat_messages`, `interview_results`,
`quiz_results`) ran as three sequential `await`s. Changed to
`Promise.all([...])` — same three round trips, concurrent instead of
serialized.

### Interview start: independent lookups serialized
`routes/interview.js`

`getActiveInterview(sessionId)` (resume check) shares no data with
`getUserProfileContext(req.user.id)` (personalization), but the resume
lookup didn't start until the profile fetch fully completed. Kicked off the
resume lookup (`activeRecordPromise`) before the profile fetch, `await`ed
where it's actually used later — overlaps the two round trips.

### Detached DB writes with no error handling (crash risk, not just perf)
`routes/chat.js` — `saveChatMessage(...)` called at 4 sites without
`await` and without `.catch()`. `saveChatMessage` throws on a Supabase
error (`lib/db.js`), so a transient DB blip became an `unhandledRejection` —
under Node's default policy, an unhandled rejection can crash the process
the same way the sweeper's `ReferenceError` did. Added `.catch(err => ...)`
at all four sites, plus a process-level `unhandledRejection` logger in
`server.js` as a backstop for anything else with the same shape.

## Not yet applied — larger changes, flagged for a follow-up pass

These came out of the same audit but weren't applied because they're bigger,
riskier, or need a product decision (not just a mechanical fix):

- **`bcryptjs` (pure JS) at cost 12** blocks the event loop for ~250-400ms
  per login/signup — the whole server stalls during that window, including
  live voice sessions. Switching to native `bcrypt` or `argon2` moves the
  hash off the main thread onto libuv's threadpool. Same API, drop-in.
- **Missing composite DB indexes.** Several queries filter on an equality
  column and sort by `created_at`, but only the equality column is indexed
  (e.g. `chat_messages(document_id, created_at)`,
  `documents(user_id, created_at desc)`). Postgres sorts the full matching
  set on every call. A few of these queries also have no `.limit()` at all
  (`getUserDocuments`, `getInterviewResults`, `getQuizResults`) — unbounded
  result sets that grow with the user's history.
- **O(n²) DOM re-parse on streamed tokens**, `public/index.html`:
  `bubble.innerHTML = fullText.replace(...)` re-parses the *entire* growing
  answer on every single streamed token instead of appending a text node.
  Visible as jank on long answers.
- **`ragCache` in `lib/interview/interviewAgent.js`** has no TTL or size cap,
  and its key (`sessionId:query.slice(0,40)`) can collide between different
  queries that share a 40-char prefix.
- **`activeVoiceAgents` (conversational-AI path) can leak independently of
  `sessionCache`** — when a session has no associated document,
  `ensureSession` never populates `sessionCache[sessionId]`, so the TTL
  sweeper (which iterates `sessionCache`) never sees that agent. A user who
  closes the tab without calling `/conversational-ai/stop` leaks a live
  LiveKit room + Gemini Live socket indefinitely. Needs sweeping
  `activeAgents`/`activeConversationalAgents` independently, keyed on their
  own `startedAt`, not piggybacked on `sessionCache` iteration.
- **Image chat re-fetches and re-uploads the full base64 image** from Qdrant
  on every turn (`lib/pipeline/imageChat.js`) instead of caching it on the
  session object for the session's lifetime.
- **Query classifier blocks retrieval** in `lib/pipeline/queryRouter.js` —
  `hybridRetrieve` doesn't start until the classifier LLM call finishes, even
  though retrieval only needs the classifier's output for query refinement,
  not to begin at all.
