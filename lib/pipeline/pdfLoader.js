import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { pipelineLog } from "../logger.js";

export async function extractTextFromPDF(filePath) {
  const start = performance.now();
  pipelineLog.info({ filePath }, 'PDF extraction started');

  try {
    const loader = new PDFLoader(filePath, {
      splitPages: true,
    });

    // Returns an array of LangChain Document objects
    const docs = await loader.load();
    const durationMs = Math.round(performance.now() - start);
    const totalChars = docs.reduce((sum, d) => sum + (d.pageContent?.length || 0), 0);

    pipelineLog.info(
      { filePath, pages: docs.length, totalChars, durationMs },
      'PDF extraction complete'
    );

    return docs;
  } catch (error) {
    const durationMs = Math.round(performance.now() - start);
    pipelineLog.error(
      { filePath, err: error.message, durationMs },
      'PDF extraction failed'
    );
    throw error;
  }
}
