import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

export async function extractTextFromPDF(filePath) {
  try {
    const loader = new PDFLoader(filePath, {
      splitPages: true,
    });

    // Returns an array of LangChain Document objects
    const docs = await loader.load();
    return docs;
  } catch (error) {
    console.error("Error reading PDF with LangChain:", error);
    throw error;
  }
}
