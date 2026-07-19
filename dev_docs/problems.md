# Development Problems and Solutions

This document outlines the various issues encountered while running the initial RAG (Retrieval-Augmented Generation) pipeline and how they were resolved.

## 1. Import Path Errors for Chains
* **Error**: `ERR_PACKAGE_PATH_NOT_EXPORTED` when importing `createStuffDocumentsChain` and `createRetrievalChain`.
* **Cause**: LangChain frequently restructures its packages across versions. The chain functions were no longer available under `@langchain/core/chains/...` or the base `langchain/chains/...` in the version we were targeting (v1.x).
* **Solution**: Updated the imports to use the `@langchain/classic` package, where legacy chain architectures have been moved.
  * *New paths:* `@langchain/classic/chains/combine_documents` and `@langchain/classic/chains/retrieval`.

## 2. Deprecated Embedding Model (Gemini API)
* **Error**: `ChromaValueError: Expected each embedding to be a non-empty array of numbers...` and later `[404 Not Found] models/text-embedding-004 is not found`.
* **Cause**: The codebase attempted to use `text-embedding-004`, an outdated model identifier that is no longer supported by the Gemini API endpoints. When the API failed silently initially, it resulted in empty embeddings being sent to ChromaDB.
* **Solution**: Updated `lib/embeddings.js` to use the currently supported model identifier: `gemini-embedding-001`.

## 3. ChromaDB Strict Metadata Formatting
* **Error**: `ChromaValueError: Expected metadata value for key 'pdf' to be a string, number, boolean, SparseVector, typed array...`
* **Cause**: LangChain's `PDFLoader` automatically captures metadata from the document and adds it to the chunk. In this case, an object structure under the `pdf` key was passed. However, ChromaDB v3 has strict formatting rules and only accepts flat primitive data values (strings, numbers, booleans) for metadata fields.
* **Solution**: Added a sanitization step in `lib/vectorStore.js` before inserting the documents into ChromaDB. The code maps through existing chunks and removes any nested objects from the `metadata`, retaining only pure primitive key-value pairs.

## 4. API Key Resolution for the Chat Model
* **Error**: `Error: Please set an API key for Google GenerativeAI in the environment variable GOOGLE_API_KEY...`
* **Cause**: By default, the `ChatGoogleGenerativeAI` LangChain constructor searches the environment for `GOOGLE_API_KEY`. Our project was utilizing `GEMINI_API_KEY` defined in the `.env`.
* **Solution**: Explicitly configured the constructor in `lib/rag.js` to map to our variable: `apiKey: process.env.GEMINI_API_KEY`.

## 5. LangChain Package Version Mismatch
* **Error**: `TypeError: text.replace is not a function` occurring exactly during the chunk retrieval phase.
* **Cause**: There was a significant version mismatch between the installed `langchain` package (`0.2.20`) and the newer suite of tools (`@langchain/core@1.1.29` and `@langchain/google-genai@2.1.22`). Due to this version spread, the underlying Retriever was passing a complex `Document` object to the embedding function during the similarity search rather than a plain string, causing standard string operations like `.replace()` to crash.
* **Solution**: Upgraded the `langchain` package to match the modern ecosystem (`npm install langchain@latest --legacy-peer-deps`), bringing it from `0.2.x` to `1.2.x` and eliminating the mismatch.

## 6. Qdrant Cloud False Alarm — Wildcard DNS Masked a Typo'd URL
* **Symptom**: Upload flow returned 500 with `err: "Not Found"` right after a successful embedding step, preceded by `Failed to obtain server version. Unable to check client-server compatibility.` A `curl` test against the cluster URL (retyped by hand rather than read from `.env`) returned `404 page not found` on every path — root, `/collections`, with and without `:6333`, with and without the API key.
* **Wrong diagnosis reached first**: the 404 response looked like a generic Go `net/http` default handler (plain text, `content-length: 19`) rather than Qdrant's own JSON error format, which was read as "the cluster has been paused or deleted."
* **Actual cause**: `*.aws.cloud.qdrant.io` is wildcard DNS behind a shared proxy — a mistyped cluster ID still resolves to a real host and gets the generic 404 from the proxy, not a Qdrant-specific error. The retyped URL simply had the wrong cluster ID. The cluster itself was, and had been, healthy the whole time.
* **How it was actually resolved**: re-ran the same `curl` test but read `QDRANT_URL`/`QDRANT_API_KEY` directly from `.env` instead of retyping them — `/collections` returned `200` with 60 real collections. `storeDocuments()` and the full upload pipeline (PDF buffer → chunks → embed → Qdrant → similarity search) were then re-run end-to-end and succeeded.
* **Lesson**: when a cloud service's DNS is wildcarded, "everything 404s" does not imply "the resource is gone" — it can just as easily mean the hostname is wrong. Always read connection strings for a live test from the actual config file (`.env`, secrets store, etc.) rather than transcribing them by hand; a single mistyped character in a subdomain can silently resolve to a *different, real* endpoint instead of failing to resolve at all.
* **Residual possibility**: the original 500 may still have been a real, transient event (e.g. a free-tier cold start — the log shows a version-compatibility check failing right before the "Not Found", consistent with the cluster waking up mid-request). If it recurs, the fix is a retry-with-backoff around the Qdrant store step in `lib/pipeline/vectorStore.js`, not further URL debugging.
