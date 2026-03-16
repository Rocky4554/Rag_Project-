import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

function normalizeForDedupe(text) {
  return text
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function isLowValueChunk(text) {
  const t = text.replace(/\s+/g, " ").trim();
  if (t.length < 80) return true;
  if (/^page\s+\d+(\s+of\s+\d+)?$/i.test(t)) return true;
  if (/^(copyright|all rights reserved)/i.test(t)) return true;
  if (/^table of contents$/i.test(t)) return true;
  return false;
}

export async function splitText(documents, chunkSize = 900, overlap = 150) {
  // Recursive splitter preserves semantic boundaries while controlling chunk size.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: chunkSize,
    chunkOverlap: overlap,
  });

  const splitDocs = await splitter.splitDocuments(documents);

  const seen = new Set();
  const validDocs = [];
  let lowValueDropped = 0;
  let dedupDropped = 0;

  for (const doc of splitDocs) {
    const content = doc.pageContent || "";
    if (isLowValueChunk(content)) {
      lowValueDropped++;
      continue;
    }

    const key = normalizeForDedupe(content);
    if (!key || seen.has(key)) {
      dedupDropped++;
      continue;
    }
    seen.add(key);

    // Keep small helpful metadata for smarter retrieval/debugging.
    doc.metadata = {
      ...(doc.metadata || {}),
      chunkChars: content.length,
    };
    validDocs.push(doc);
  }

  console.log(
    `Split ${splitDocs.length} chunks -> kept ${validDocs.length} | dropped low-value: ${lowValueDropped} | dropped duplicate: ${dedupDropped}.`
  );

  return validDocs;
}