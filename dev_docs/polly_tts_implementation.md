# Amazon Polly TTS — Swappable Provider Architecture

## Goal

Replace Kokoro (Docker-based CPU TTS) with **Amazon Polly** (cloud API TTS), structured using a **Provider Pattern** so you can swap TTS engines in the future by changing a single env variable (`TTS_PROVIDER`).

---

## Architecture: The Provider Pattern

```
lib/tts/
  ├── index.js              ← Entry point — reads TTS_PROVIDER, loads the right provider
  ├── providers/
  │   ├── pollyProvider.js  ← Amazon Polly implementation  [DEFAULT]
  │   └── kokoroProvider.js ← Kokoro-FastAPI implementation [LEGACY]
```

The interview agent and summary feature **never call Polly or Kokoro directly**.  
They only ever call `textToAudio(text)` from `lib/tts/index.js`.  
Swapping providers = change `.env` only. Zero code changes.

```
TTS_PROVIDER=polly    → uses pollyProvider.js
TTS_PROVIDER=kokoro   → uses kokoroProvider.js (fallback, Docker still needed)
```

---

## Files to Change

### [NEW] `lib/tts/providers/pollyProvider.js`

- Uses `@aws-sdk/client-polly`
- Reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` from `.env`
- Splits text into ≤3000 char chunks (Polly API limit per request)
- Calls `SynthesizeSpeechCommand` for each chunk using **Neural engine**
- Default voice: `Joanna` (configurable via `POLLY_VOICE_ID` env var — easy to swap)
- Merges binary MP3 buffers → returns single base64 string

```js
// Swappable voice without code changes:
VoiceId: process.env.POLLY_VOICE_ID || "Joanna"
```

### [NEW] `lib/tts/providers/kokoroProvider.js`

- Moves existing logic from `lib/speechToAudio.js` exactly as-is
- Keeps old retry logic and chunking from previous work

### [NEW] `lib/tts/index.js`  ← The key file

```js
// Reads TTS_PROVIDER env var — defaults to "polly"
const provider = process.env.TTS_PROVIDER || "polly";

const providers = {
  polly: () => import("./providers/pollyProvider.js"),
  kokoro: () => import("./providers/kokoroProvider.js"),
};

export async function textToAudio(text) {
  const mod = await providers[provider]?.();
  if (!mod) throw new Error(`Unknown TTS provider: ${provider}`);
  return mod.textToAudio(text);
}
```

### [MODIFY] `lib/speechToAudio.js`

- Becomes a thin re-export wrapper pointing to `lib/tts/index.js`
- So existing imports in `server.js` keep working with zero changes

```js
// lib/speechToAudio.js (after change)
export { textToAudio } from "./tts/index.js";
```

### [MODIFY] `.env` / `.env.example`

Add:
```
TTS_PROVIDER=polly
POLLY_VOICE_ID=Joanna
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

---

## New npm Dependency

```bash
npm install @aws-sdk/client-polly
```

That's the **only** new package. No Docker changes needed for Polly.

---

## How to Switch TTS in the Future

| What to do | How |
|---|---|
| Use Polly | `TTS_PROVIDER=polly` in `.env` |
| Use Kokoro | `TTS_PROVIDER=kokoro` in `.env` (keep Docker running) |
| Add a new provider (e.g. ElevenLabs) | 1. Create `lib/tts/providers/elevenlabsProvider.js` — 2. Add `elevenlabs` key to the `providers` map in `index.js` — 3. Set `TTS_PROVIDER=elevenlabs` |

No code changes in `server.js`, `interviewAgent.js`, or `summarizer.js` — ever.

---

## Verification Plan

1. Ensure `.env` has valid AWS credentials and `TTS_PROVIDER=polly`
2. Run `node server.js`
3. Upload a PDF → click **🔊 Summarize PDF**
4. ✅ Audio plays using Polly's voice (no Docker Kokoro container needed)
5. Change `TTS_PROVIDER=kokoro` → restart → summarize again
6. ✅ Falls back to Kokoro — confirms provider swap works

### AWS Quick Test (optional terminal check):
```bash
node -e "
import('@aws-sdk/client-polly').then(({ PollyClient, SynthesizeSpeechCommand }) => {
  const c = new PollyClient({ region: 'ap-south-1' });
  c.send(new SynthesizeSpeechCommand({ Text: 'Hello!', OutputFormat: 'mp3', VoiceId: 'Joanna', Engine: 'neural' }))
   .then(r => console.log('Polly OK:', r.ContentType))
   .catch(e => console.error('Polly Error:', e.message));
});
"
```
