# Voice Agent "Not Starting" — Orphaned STT Loop + Session Sweeper Crash

## Symptom
On the live site, starting an interview produced no audio and no visible error.
Production logs showed an infinite loop with no other agent activity mixed in:

```
STT connecting to Deepgram
STT Deepgram WebSocket opened
STT Deepgram WebSocket closed   code: 1011   reason: "Deepgram did not
  receive audio data or a text message within the timeout window." (NET-0001)
STT reconnecting in 1s
```

repeating every ~3-8s, forever, for a single session. No `LiveKit room connected`,
no `Subscribed to user audio`, no transcript ever appeared.

## Root cause #1 — orphaned STT connection on failed LiveKit join

`VoicePipelineWorker.start()` (`agents/shared/voicePipelineWorker.js`) starts
Deepgram STT as step 1, *before* it ever touches LiveKit:

```js
async start() {
    this.stt.start();              // step 1 — always runs
    const token = await this.generateToken();
    await this.room.connect(...);  // step 3 — can fail
    ...
}
```

If the LiveKit room connection then failed (all 3 internal retries exhausted),
`start()` threw — but **nothing stopped the STT it had already started**.
`routes/interview.js` retried `agent.start()` once more after 2s, and if that
*also* failed, it just logged the error and gave up. The `DeepgramSTT`
instance kept reconnecting to Deepgram forever with no LiveKit room ever
backing it — that's the infinite loop in the logs. The loop was a symptom,
not the disease: the agent had already failed to start; only its STT half
was still visibly running.

One specific LiveKit failure observed in production:
```
err: "engine: signal failure: client error: 401 Unauthorized - invalid token: revoked"
```
LiveKit revokes a token when a second connection joins a room with the same
participant identity. `InterviewAgentWorker` uses a fixed identity
(`"ai-interviewer"`), so a stale connection from a worker that died without
disconnecting (see root cause #2) can collide with the next attempt for the
same session/room.

### Fix
`agents/shared/voicePipelineWorker.js` — wrapped steps 2-4 of `start()` in
try/catch. On any failure it now calls `this.stt.stop()` and
`this.room.disconnect()` before rethrowing, so a failed startup never leaves
a live component behind. Also disconnects a failed room attempt before
recreating it on retry (previously `this.room = new Room()` replaced the
reference without disconnecting the old one first).

`routes/interview.js` — on final retry failure, removes the dead agent from
`activeAgents` (so a future attempt for the same session isn't blocked) and
emits `interview_error` over Socket.io. `public/index.html` listens for it
and surfaces the message via `ivSetState`, instead of leaving the UI
silently stuck.

## Root cause #2 — session sweeper crashed the whole process

`server.js` declares the voice-agent map as `activeConversationalAgents`
(line 51), but the 10-minute TTL sweeper referenced `activeVoiceAgents`
(never declared anywhere as a variable — it only exists as a *key* in the
`deps` object passed to route factories):

```js
setInterval(() => {
    for (const [id, session] of Object.entries(sessionCache)) {
        if (now - session.createdAt > SESSION_TTL) {
            if (activeVoiceAgents.has(id)) { ... }   // ReferenceError
```

The moment any session exceeded `SESSION_TTL` (2h default), this threw
`ReferenceError: activeVoiceAgents is not defined` inside a `setInterval`
callback — an uncaught exception that **killed the Node process**. Two
compounding effects:

1. The sweep aborted before `delete sessionCache[id]`, so the code written
   to prevent a memory leak was itself causing one (session cache grew
   unbounded whenever this path wasn't hit).
2. The abrupt process death didn't call `.stop()` on any live agent, leaving
   LiveKit participants connected server-side after the process was gone —
   exactly the stale-identity condition that produces the "revoked token"
   error in root cause #1 on the next deploy/restart.

### Fix
Renamed the sweeper's references to `activeConversationalAgents` and wrapped
the whole interval body in try/catch so one bad iteration can't kill the
process. Verified in isolation (mock session past TTL) that the sweeper no
longer throws and correctly evicts.

## Root cause #3 — no graceful shutdown

`server.js` had no `SIGTERM`/`SIGINT` handler at all. On a Render redeploy,
the old container is killed outright — any in-progress interview's LiveKit
participant is abandoned mid-connection rather than disconnected, which is
the other path into the stale-identity/revoked-token condition above.

Complicating this further: `lib/logger.js` registered the *only* `SIGTERM`
handler in the codebase (conditional on OTLP logging being enabled) and
called `process.exit(0)` immediately. Since `logger.js` is imported before
`server.js` runs its own setup, that handler would fire first and hard-exit
before any agent cleanup could happen — even if `server.js` added its own
handler naively.

### Fix
- `server.js`: added a `shutdown(signal)` function that stops every agent in
  both `activeAgents` and `activeConversationalAgents` (each wrapped in its
  own try/catch so one throwing `.stop()` can't block the rest), clears both
  maps, closes the HTTP server, and force-exits after an 8s backstop.
  Registered on both `SIGTERM` and `SIGINT`, with a `shuttingDown` guard so
  a second signal doesn't re-enter.
- `lib/logger.js`: the OTLP `SIGTERM`/`SIGINT` handlers now only flush
  buffered logs — they no longer call `process.exit()`, so `server.js`'s
  handler is free to run to completion.

### Platform note
`SIGTERM` cannot be delivered to a Node process on Windows — verified with a
minimal repro (a bare `process.on('SIGTERM', ...)` handler never fires when
the process is killed via `child.kill('SIGTERM')` on `win32`). The shutdown
logic itself was verified directly (unit-style, against mock agent objects)
rather than via a live signal test. It targets the Linux container in
production, where `SIGTERM` delivery is normal.

## Verification
- Reproduced the sweeper crash in isolation before fixing (bare repro of the
  `ReferenceError`), and confirmed the fixed version evicts without throwing.
- Verified `shutdown()` logic: a throwing agent's `.stop()` doesn't prevent
  the others from stopping, both maps end up empty, and a second call is a
  no-op.
- Server boots clean end-to-end with all changes applied.

## Related
See `livekit_connectivity_fix.md` for a different root cause of the same
`engine: signal failure` error family (missing CA certificates in the
production Docker image) — "signal failure" is LiveKit's generic
`@livekit/rtc-node` error wrapper and can come from several distinct causes.
Don't assume they're the same bug just because the log prefix matches.
