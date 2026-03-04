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
