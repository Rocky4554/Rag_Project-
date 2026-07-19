import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { pipelineLog } from "../logger.js";

/**
 * Extract text from a PDF.
 *
 * @param {string|Buffer} source - A file path, or the PDF bytes themselves.
 *   Passing a Buffer avoids a pointless write-to-disk/read-back round trip
 *   when the file is already in memory (multer memory storage).
 */
export async function extractTextFromPDF(source) {
  const start = performance.now();
  const filePath = Buffer.isBuffer(source) ? '<buffer>' : source;
  pipelineLog.info({ filePath }, 'PDF extraction started');

  try {
    const input = Buffer.isBuffer(source)
      ? new Blob([source], { type: 'application/pdf' })
      : source;

    const loader = new PDFLoader(input, {
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
