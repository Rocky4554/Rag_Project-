import { Router } from 'express';
import { registerVectorStore, parseTTSResponse } from "../lib/interview/interviewAgent.js";
import { InterviewAgentWorker } from "../agent/agent.js";
import { optionalAuth } from "../middleware/auth.js";
import { getDocumentBySessionId, saveInterviewResult, logActivity } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { updateUserProfileAfterInterview, getUserProfileContext } from "../lib/interview/profileUpdater.js";
import { validate, interviewStartSchema } from '../lib/validation.js';
import { interviewLog } from '../lib/logger.js';

export function createInterviewRoutes({ sessionCache, activeAgents, clientReadyResolvers, io, interviewAgent }) {
    const router = Router();

    router.post('/interview/start', validate(interviewStartSchema), optionalAuth, async (req, res) => {
        try {
            const { sessionId, maxQuestions = 5 } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Session not found. Please upload the PDF again.' });
            }

            interviewLog.info({ sessionId, maxQuestions }, 'Starting interview');

            registerVectorStore(sessionId, session.vectorStore);

            // Fetch cross-session user profile for personalized interview
            let userProfileContext = "";
            if (req.user) {
                try {
                    const profileCtx = await getUserProfileContext(req.user.id);
                    if (profileCtx) {
                        userProfileContext = profileCtx;
                        interviewLog.info({ sessionId }, 'Injecting user profile context into interview');
                    }
                } catch (err) {
                    interviewLog.warn({ err: err.message }, 'Failed to load user profile context');
                }
            }

            const h = new Date().getHours();
            const timeGreeting = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';

            const initialState = {
                sessionId,
                maxQuestions: parseInt(maxQuestions),
                difficultyLevel: "medium",
                chatHistory: [],
                questionsAsked: 0,
                topicsUsed: [],
                userProfileContext,
                candidateName: req.user?.name || "there",
                timeGreeting,
            };

            // thread_id for LangSmith tracing + PostgresSaver checkpointing
            const config = { configurable: { thread_id: sessionId } };

            // Kill previous agent if exists
            if (activeAgents.has(sessionId)) {
                activeAgents.get(sessionId).stop();
            }

            const agent = new InterviewAgentWorker(sessionId, sessionCache, interviewAgent, io);
            activeAgents.set(sessionId, agent);

            // Store save callback so the agent worker can save results when interview ends
            session._onInterviewComplete = async (finalState) => {
                if (req.user) {
                    try {
                        const doc = await getDocumentBySessionId(sessionId);
                        if (doc) {
                            await saveInterviewResult({
                                userId: req.user.id,
                                documentId: doc.id,
                                threadId: sessionId,
                                questionsAsked: finalState.questionsAsked,
                                scores: finalState.scores,
                                topicScores: finalState.topicScores,
                                finalReport: finalState.finalReport,
                                difficultyLevel: finalState.difficultyLevel
                            });
                            logActivity({
                                userId: req.user.id,
                                action: 'interview_completed',
                                metadata: { sessionId, questionsAsked: finalState.questionsAsked }
                            });

                            // Update user profile with cross-session memory (fire-and-forget)
                            updateUserProfileAfterInterview(req.user.id, finalState, doc.original_name)
                                .catch(err => interviewLog.error({ err: err.message }, 'Profile update failed'));
                        }
                    } catch (err) {
                        interviewLog.error({ err: err.message, sessionId }, 'Failed to save interview result');
                    }
                }
            };

            // === PARALLEL: Run LangGraph invoke + agent.start() concurrently ===
            const [resultState] = await Promise.all([
                interviewAgent.invoke(initialState, config),
                agent.start()
            ]);

            session.interviewStateConfig = config;

            // Log activity if user is authenticated
            if (req.user) {
                logActivity({
                    userId: req.user.id,
                    action: 'interview_started',
                    metadata: { sessionId, maxQuestions: parseInt(maxQuestions) }
                });
            }

            // Agent is already connected — speak first question immediately
            const { uniquePart } = parseTTSResponse(resultState.currentQuestion);
            io.to(sessionId).emit('transcript_final', { role: 'ai', text: uniquePart });

            // Wait for client audio ready (reduced from 10s to 2s)
            interviewLog.info({ sessionId }, 'Waiting for client audio ready');
            await new Promise((resolve) => {
                clientReadyResolvers.set(sessionId, resolve);
                setTimeout(() => {
                    if (clientReadyResolvers.has(sessionId)) {
                        interviewLog.warn({ sessionId }, 'client_audio_ready timeout — speaking anyway');
                        clientReadyResolvers.delete(sessionId);
                        resolve();
                    }
                }, 2000);
            });

            interviewLog.info({ sessionId }, 'Speaking first question');
            // Fire-and-forget — don't block the response
            agent.speak("Hello! Welcome to your AI voice interview. " + uniquePart).catch(err => {
                interviewLog.error({ err: err.message, sessionId }, 'Agent speak error');
            });

            res.json({
                questionNumber: resultState.questionsAsked,
                difficulty: resultState.difficultyLevel,
                agentStarted: true
            });

        } catch (error) {
            interviewLog.error({ err: error.message }, 'Interview start error');
            res.status(500).json({ error: error.message || "Failed to start interview" });
        }
    });

    return router;
}
