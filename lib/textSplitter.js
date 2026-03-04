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
  return splitDocs;
}