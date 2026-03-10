import dotenv from 'dotenv';
dotenv.config();
import { storeDocuments } from "./lib/vectorStore.js";

async function main() {
    try {
        console.log("Testing storeDocuments with a dummy document...");
        const docs = [{
            pageContent: "This is a test document.",
            metadata: { source: "test" }
        }];
        await storeDocuments(docs, "test_collection");
        console.log("Success!");
    } catch (err) {
        console.error("storeDocuments error:", err.message);
    }
}
main();
