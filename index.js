import { extractTextFromPDF } from "./lib/pdfLoader.js";
import { splitText } from "./lib/textSplitter.js";
import { storeDocuments } from "./lib/vectorStore.js";
import { queryRAG } from "./lib/rag.js";

async function run() {
  const text = await extractTextFromPDF("./sample.pdf");

  const docs = await splitText(text);

  const vectorStore = await storeDocuments(docs);

  const answer = await queryRAG(
    vectorStore,
    "Tell me what type of project i can add in this resume"
  );

  console.log(answer);
}

run();