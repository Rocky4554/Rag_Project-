# PDF Summarization + TTS Feature — Implementation Plan

Add a **"Summarize PDF"** button that generates an AI summary of the uploaded PDF and plays it as audio using **Kokoro-FastAPI** (self-hosted TTS via Docker). The feature slots in as **Phase 3** after upload, reusing docs already in memory — no re-parsing needed.

---

## Architecture Overview

```
POST /api/upload  →  PDFLoader → Splitter → ChromaDB  ✅ (already works)
                                    ↓
                         docs cached in sessionCache
                                    ↓
POST /api/quiz    →  ChromaDB → Gemini → JSON quiz     ✅ (already works)
                                    ↓
POST /api/summary →  cached docs → Gemini → summary text
                                           → Kokoro-FastAPI → base64 MP3  🆕
```

---

## Files to Change

### [NEW] `lib/summarizer.js`

Two exported functions:

**`summarizeDocs(docs)`**
- Uses `createStuffDocumentsChain` + `ChatPromptTemplate` — same pattern as `quizGenerator.js` (NOT the deprecated `loadSummarizationChain`)
- Sends all PDF chunks to Gemini with a prompt to produce a concise, readable summary
- Returns a plain text string

**`textToAudio(text)`**
- Calls Kokoro-FastAPI at `KOKORO_API_URL` env var (e.g. `http://localhost:8880`)
- Splits long text into sentence-based chunks (max 4800 chars each)
- Fetches MP3 from each chunk, collects as `Buffer` objects
- Uses `Buffer.concat(buffers)` to merge correctly (**not** base64 string join — that corrupts the file)
- Returns single base64-encoded MP3 string

```js
// Kokoro API call (OpenAI-compatible endpoint)
POST {KOKORO_API_URL}/v1/audio/speech
Body: { model: "kokoro", input: textChunk, voice: "af_heart", response_format: "mp3" }
```

---

### [MODIFY] `server.js`

**Change 1 — Cache raw docs alongside vectorStore:**
```js
// After extractTextFromPDF + splitText, store both:
sessionCache[sessionId] = {
  vectorStore: vectorStore,
  docs: docs          // ← add this (raw LangChain Document array)
};
```

**Change 2 — Add new route:**
```js
app.post('/api/summary', async (req, res) => {
  const { sessionId } = req.body;
  const session = sessionCache[sessionId];
  if (!session) return res.status(404).json({ error: 'Session not found' });

  const summary = await summarizeDocs(session.docs);
  const audio = await textToAudio(summary);   // base64 MP3

  res.json({ summary, audio });
});
```

> ⚠️ `sessionCache` currently stores the vectorStore directly (not an object). This change wraps it in `{ vectorStore, docs }` — also update the `/api/quiz` route to use `session.vectorStore`.

---

### [MODIFY] `public/index.html`

Add a **Summary card** that appears after upload success (alongside the existing config section):

**HTML to add:**
```html
<!-- Summary Section -->
<div class="card hidden" id="summary-section">
  <h2 style="margin-top:0">PDF Summary</h2>
  <button id="summary-btn" onclick="generateSummary()">🔊 Summarize PDF</button>
  <div id="summary-loader" class="loader hidden"></div>
  <div id="summary-status" class="status-msg"></div>
  <audio id="summary-audio" controls style="display:none; width:100%; margin-top:1rem"></audio>
  <p id="summary-text" style="margin-top:1rem; line-height:1.8; color:var(--text-secondary)"></p>
</div>
```

**JS to add:**
```js
async function generateSummary() {
  const btn = document.getElementById('summary-btn');
  const loader = document.getElementById('summary-loader');
  const status = document.getElementById('summary-status');

  btn.disabled = true;
  loader.classList.remove('hidden');
  status.textContent = 'Generating summary and audio...';

  try {
    const res = await fetch('/api/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId })   // sessionId already in scope
    });
    const { summary, audio } = await res.json();

    document.getElementById('summary-text').textContent = summary;

    const audioEl = document.getElementById('summary-audio');
    audioEl.src = `data:audio/mp3;base64,${audio}`;
    audioEl.style.display = 'block';
    status.textContent = '';
  } catch (err) {
    status.textContent = err.message;
    status.className = 'status-msg error-msg';
  } finally {
    btn.disabled = false;
    loader.classList.add('hidden');
  }
}
```

Reveal the section after upload succeeds (same place `config-section` is revealed):
```js
document.getElementById('summary-section').classList.remove('hidden');
```

---

### [NEW] `docker-compose.yml` (project root — for local dev)

```yaml
version: "3.9"
services:
  app:
    build: .
    ports:
      - "3000:3000"
    env_file: .env
    depends_on:
      - chromadb
      - kokoro

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma   # persistent storage

  kokoro:
    image: ghcr.io/remsky/kokoro-fastapi-cpu:latest
    ports:
      - "8880:8880"

volumes:
  chroma_data:
```

---

### [NEW] `railway.toml` (project root — for Railway deployment)

```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "node server.js"
healthcheckPath = "/"
restartPolicyType = "ON_FAILURE"
```

> On Railway: add ChromaDB and Kokoro as separate Docker services in the same project. Railway generates internal URLs (e.g. `kokoro.railway.internal:8880`). Set these as environment variables.

---

### [NEW] `.env.example`

```
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_URL=http://localhost:8000
KOKORO_API_URL=http://localhost:8880
PORT=3000
```

---

## Installation

```bash
# One new npm dependency (for the HTTP call to Kokoro-FastAPI)
# Actually no new dependency needed — use Node.js built-in fetch (Node 18+)

# Start local services (requires Docker)
docker compose up -d

# Start app
node server.js
```

---

## Verification Steps

### Local
1. `docker compose up -d` → start ChromaDB + Kokoro
2. `node server.js` → start Express server
3. Open `http://localhost:3000`
4. Upload a PDF → wait for processing
5. Click **"🔊 Summarize PDF"**
6. ✅ Summary text appears + audio player loads and speaks the summary

### API Test
```bash
# After uploading a PDF and getting sessionId from /api/upload response:
curl -X POST http://localhost:3000/api/summary \
  -H "Content-Type: application/json" \
  -d "{\"sessionId\": \"<your-sessionId>\"}"

# Expected: { "summary": "...", "audio": "<base64 string>" }
```

### Regression Check
- Upload PDF → generate quiz → submit answers → all existing behaviour unchanged
- `server.js` changes are additive; only `sessionCache` structure changes internally

---

## Railway Deployment Checklist

- [ ] Push code to GitHub
- [ ] Create Railway project → Deploy from GitHub (Node.js app)
- [ ] Add Docker service → `chromadb/chroma` → attach a Volume at `/chroma/chroma`
- [ ] Add Docker service → `ghcr.io/remsky/kokoro-fastapi-cpu:latest`
- [ ] Set env vars: `GEMINI_API_KEY`, `CHROMA_URL` (Railway internal URL), `KOKORO_API_URL` (Railway internal URL)
- [ ] Deploy → test live URL
