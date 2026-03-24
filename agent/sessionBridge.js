import { interviewStateChannels } from "../lib/interview/interviewAgent.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { agentLog } from "../lib/logger.js";

/**
 * Acts as the bridge between the real-time LiveKit flow and your existing
 * stateless/LangGraph `interviewAgent.js` logic.
 * Emulates the behavior of `POST /api/interview/answer`.
 */
export class SessionBridge {
    constructor(sessionId, sessionCache, agentWorkflow) {
        this.sessionId = sessionId;
        this.sessionCache = sessionCache;
        this.agentWorkflow = agentWorkflow;
    }

    /**
     * Called when the STT emits a final user transcript.
     * Invokes your LangGraph agent directly.
     */
    async processUserTranscript(transcript) {
        const session = this.sessionCache[this.sessionId];
        if (!session || !session.interviewStateConfig) {
            throw new Error("Session not found or interview not started.");
        }

        const start = performance.now();
        agentLog.info({ sessionId: this.sessionId, transcriptLength: transcript.length }, 'SessionBridge processing transcript');

        const inputState = { userAnswer: transcript };
        const resultState = await this.agentWorkflow.invoke(inputState, session.interviewStateConfig);
        const durationMs = Math.round(performance.now() - start);
        agentLog.info({ sessionId: this.sessionId, durationMs, done: !!resultState.finalReport }, 'SessionBridge invoke complete');

        const isDone = !!resultState.finalReport;
        
        return {
            done: isDone,
            evaluation: resultState.evaluation,
            nextQuestion: isDone ? null : resultState.currentQuestion,
            finalReport: resultState.finalReport,
            difficulty: resultState.difficultyLevel,
            questionNumber: resultState.questionsAsked,
            // Pass through for DB persistence
            questionsAsked: resultState.questionsAsked,
            scores: resultState.scores,
            topicScores: resultState.topicScores,
            difficultyLevel: resultState.difficultyLevel
        };
    }
}