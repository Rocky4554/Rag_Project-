import { interviewStateChannels } from "../../lib/interview/interviewAgent.js";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { agentLog } from "../../lib/logger.js";
import { upsertActiveInterview } from "../../lib/db.js";

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
     * acousticMeta = { utteranceDurationMs, fillerWordCount, timeToAnswer, bargedIn }
     * Invokes your LangGraph agent directly.
     */
    async processUserTranscript(transcript, acousticMeta = {}) {
        const session = this.sessionCache[this.sessionId];
        if (!session || !session.interviewStateConfig) {
            throw new Error("Session not found or interview not started.");
        }

        const start = performance.now();
        agentLog.info({
            sessionId: this.sessionId,
            transcriptLength: transcript.length,
            utteranceDurationMs: acousticMeta.utteranceDurationMs || 0,
            fillerWordCount: acousticMeta.fillerWordCount || 0,
            timeToAnswer: acousticMeta.timeToAnswer || 0,
            bargedIn: acousticMeta.bargedIn || false,
        }, 'SessionBridge processing transcript');

        // Improvement #4: inject acoustic behavioral context into LangGraph state
        const inputState = {
            userAnswer: transcript,
            utteranceDurationMs: acousticMeta.utteranceDurationMs || 0,
            fillerWordCount: acousticMeta.fillerWordCount || 0,
            timeToAnswer: acousticMeta.timeToAnswer || 0,
            bargedIn: acousticMeta.bargedIn || false,
        };
        const resultState = await this.agentWorkflow.invoke(inputState, session.interviewStateConfig);
        const durationMs = Math.round(performance.now() - start);
        agentLog.info({ sessionId: this.sessionId, durationMs, done: !!resultState.finalReport }, 'SessionBridge invoke complete');

        const isDone = !!resultState.finalReport;

        // Keep the active interview record in sync with the latest question number
        // and current question. This way a resume always shows the right question.
        if (!isDone) {
            upsertActiveInterview({
                sessionId: this.sessionId,
                userId: null, // user_id not available here; set on start
                threadId: session.interviewStateConfig.configurable.thread_id,
                maxQuestions: resultState.maxQuestions || 5,
                questionsAsked: resultState.questionsAsked || 0,
                currentQuestion: resultState.currentQuestion || '',
            }).catch(() => {});
        }

        return {
            done: isDone,
            intent: resultState.intent,
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