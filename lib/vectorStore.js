import { Chroma } from "@langchain/community/vectorstores/chroma";
import { embedder } from "./embeddings.js";

export async function storeDocuments(docs, collectionName = "my_collection") {
    if (!docs || docs.length === 0) {
        throw new Error("No documents provided to store in ChromaDB.");
    }

    console.log(`Storing ${docs.length} document chunks in Chroma DB using LangChain...`);

    // Chroma v3 only accepts flat primitive metadata values (string, number, boolean, null).
    // PDFLoader adds nested objects (e.g. `pdf` key), so we strip those out.
    const sanitizedDocs = docs.map(doc => {
        const cleanMeta = {};
        for (const [key, value] of Object.entries(doc.metadata)) {
            if (value === null || ["string", "number", "boolean"].includes(typeof value)) {
                cleanMeta[key] = value;
            }
        }
        doc.metadata = cleanMeta;
        return doc;
    });

    // LangChain's Chroma.fromDocuments handles batching the embeddings 
    // generation and storing them in ChromaDB in a single function call!
    const vectorStore = await Chroma.fromDocuments(
        sanitizedDocs,
        embedder,
        {
            collectionName: collectionName,
            url: "http://localhost:8000", // The default Chroma compose URL
        }
    );

    return vectorStore;
}