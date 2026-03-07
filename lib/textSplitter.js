import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export async function splitText(documents, chunkSize = 1000, overlap = 200) {
  // Using RecursiveCharacterTextSplitter which splits smartly on paragraphs, then lines, then words.
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: chunkSize,
    chunkOverlap: overlap,
  });

  // Since `documents` is now an array of LangChain Document objects (from PDFLoader),
  // we use splitDocuments instead of splitText. This preserves metadata!
  const splitDocs = await splitter.splitDocuments(documents);

  // Filter out empty or near-empty chunks (< 20 chars of real content).
  // These are usually page headers, footers, or image captions that slipped through.
  // Gemini embedding API returns [] for these, which crashes ChromaDB upsert.
  const validDocs = splitDocs.filter(doc => doc.pageContent.trim().length >= 20);
  console.log(`Split into ${splitDocs.length} chunks, kept ${validDocs.length} after filtering empties.`);

  return validDocs;
}