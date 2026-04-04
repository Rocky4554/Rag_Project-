import { QdrantClient } from "@qdrant/js-client-rest";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { chatLog } from "../logger.js";
import dotenv from 'dotenv';
dotenv.config();

const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});

// Parse all available Gemini keys for rotation
const geminiKeys = (process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '')
    .split(',').map(k => k.trim()).filter(Boolean);

/**
 * Retrieve the stored image from Qdrant (base64 + metadata).
 */
async function getImageFromSession(session) {
    const collection = session.imageCollection;
    if (!collection) return null;

    try {
        const result = await qdrantClient.scroll(collection, { limit: 1, with_payload: true });
        const point = result.points?.[0];
        if (!point?.payload) return null;
        return {
            base64: point.payload.base64,
            filename: point.payload.filename,
            mimetype: point.payload.mimetype,
        };
    } catch (err) {
        chatLog.error({ err: err.message, collection }, 'Failed to retrieve image from Qdrant');
        return null;
    }
}

/**
 * Chat with an uploaded image using Gemini multimodal.
 */
export async function queryImageChat(session, question, chatHistory = []) {
    const start = performance.now();
    chatLog.info({ question: question.substring(0, 120), contentType: 'image' }, 'Image chat started');

    const imageData = await getImageFromSession(session);
    if (!imageData?.base64) {
        return "I couldn't retrieve the uploaded image. Please try re-uploading it.";
    }

    const { prompt, imageInline } = buildImagePrompt(imageData, question, chatHistory);

    // Try each Gemini key until one works
    for (let i = 0; i < geminiKeys.length; i++) {
        try {
            const genAI = new GoogleGenerativeAI(geminiKeys[i]);
            const model = genAI.getGenerativeModel({ model: "gemini-3.1-flash-lite-preview" });
            const result = await model.generateContent([prompt, imageInline]);
            const answer = result.response.text();
            const totalMs = Math.round(performance.now() - start);
            if (i > 0) chatLog.info({ key: i + 1 }, 'Image chat succeeded with fallback key');
            chatLog.info({ totalMs, answerLength: answer.length }, 'Image chat complete');
            return answer;
        } catch (err) {
            const isLast = i === geminiKeys.length - 1;
            chatLog.warn({ key: i + 1, hasNext: !isLast, err: err.message }, 'Image chat key failed');
            if (isLast) throw err;
        }
    }
}

/**
 * Build the prompt and image inline data (shared by sync + stream).
 */
function buildImagePrompt(imageData, question, chatHistory) {
    let historyContext = '';
    if (chatHistory.length > 0) {
        const recent = chatHistory.slice(-6);
        historyContext = recent.map(msg => {
            const role = msg.getType?.() === 'human' ? 'User' : 'Assistant';
            return `${role}: ${msg.content}`;
        }).join('\n') + '\n';
    }

    const prompt = `You are a helpful AI assistant analyzing an uploaded image called "${imageData.filename}".

${historyContext ? `Previous conversation:\n${historyContext}\n` : ''}User question: ${question}

Analyze the image carefully and answer the user's question. Be specific about what you see in the image. If the question is unrelated to the image, still answer helpfully but mention what the image shows.`;

    const imageInline = {
        inlineData: {
            mimeType: imageData.mimetype || 'image/png',
            data: imageData.base64,
        },
    };

    return { prompt, imageInline };
}

/**
 * Streaming version of image chat with key rotation.
 */
export async function* queryImageChatStream(session, question, chatHistory = []) {
    const start = performance.now();
    chatLog.info({ question: question.substring(0, 120), contentType: 'image', streaming: true }, 'Image chat stream started');

    const imageData = await getImageFromSession(session);
    if (!imageData?.base64) {
        yield "I couldn't retrieve the uploaded image. Please try re-uploading it.";
        return;
    }

    const { prompt, imageInline } = buildImagePrompt(imageData, question, chatHistory);

    // Try each Gemini key until one works
    for (let i = 0; i < geminiKeys.length; i++) {
        try {
            const genAI = new GoogleGenerativeAI(geminiKeys[i]);
            const model = genAI.getGenerativeModel({ model: "gemini-3.1-flash-lite-preview" });
            const result = await model.generateContentStream([prompt, imageInline]);

            let totalChars = 0;
            for await (const chunk of result.stream) {
                const text = chunk.text();
                if (text) {
                    totalChars += text.length;
                    yield text;
                }
            }

            const totalMs = Math.round(performance.now() - start);
            if (i > 0) chatLog.info({ key: i + 1 }, 'Image chat stream succeeded with fallback key');
            chatLog.info({ totalMs, totalChars, streaming: true }, 'Image chat stream complete');
            return;
        } catch (err) {
            const isLast = i === geminiKeys.length - 1;
            chatLog.warn({ key: i + 1, hasNext: !isLast, err: err.message }, 'Image chat stream key failed');
            if (isLast) throw err;
        }
    }
}
