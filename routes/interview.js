import { Router } from 'express';
import { registerVectorStore, clearSessionInterviewState, parseTTSResponse } from "../lib/interview/interviewAgent.js";
import { InterviewAgentWorker } from "../agents/interview/worker.js";
import { optionalAuth } from "../middleware/auth.js";
import { getDocumentBySessionId, saveInterviewResult, logActivity } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { updateUserProfileAfterInterview, getUserProfileContext } from "../lib/interview/profileUpdater.js";
import { validate, interviewStartSchema } from '../lib/validation.js';
import { interviewLog } from '../lib/logger.js';
import { cleanupSessionAgents } from './conversationalAI.js';

export function createInterviewRoutes({ sessionCache, activeAgents, activeVoiceAgents, clientReadyResolvers, io, interviewAgent }) {
    const router = Router();

    router.post('/interview/start', validate(interviewStartSchema), optionalAuth, async (req, res) => {
        try {
            const { sessionId, maxQuestions = 5 } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Session not found. Please upload a PDF or text document first.' });
            }
            if (session.contentType === 'image') {
                return res.status(400).json({ error: 'Interview is not available for image uploads. Please upload a PDF or text document.' });
            }

            // Diagnostic: trace exactly which document/collection we're using
            const vsCollectionName = session.vectorStore?.client?.collectionName
                || session.vectorStore?.args?.collectionName
                || session.vectorStore?.collectionName
                || 'unknown';
            interviewLog.info({
                sessionId,
                maxQuestions,
                originalName: session.originalName || 'unknown',
                contentType: session.contentType || 'unknown',
                restored: !!session._restored,
                qdrantCollection: vsCollectionName,
            }, 'Starting interview — session details');

            // Purge any stale RAG cache / vectorStore from a previous interview on this session
            clearSessionInterviewState(sessionId);
            registerVectorStore(sessionId, session.vectorStore, session.originalName || "");

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

            // Use a unique thread_id per interview run so the PostgresSaver checkpointer
            // never restores stale state (old finalReport, interviewStopped=true, old userAnswer)
            // from a previous interview on the same sessionId.
            const interviewRunId = `${sessionId}_${Date.now()}`;
            const config = { configurable: { thread_id: interviewRunId } };

            // Clean up ALL existing agents on this session before starting a new one
            await cleanupSessionAgents(sessionId, { activeAgents, activeVoiceAgents });

            const agent = new InterviewAgentWorker(
                sessionId, sessionCache, interviewAgent, io,
                initialState.candidateName,
                initialState.maxQuestions
            );
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

            // Start agent connection in background — don't block HTTP response on LiveKit
            // LiveKit room.connect() can fail transiently (region info fetch, DNS, etc.)
            // and should NOT cause the whole interview to 500.
            agent.start().catch(err => {
                interviewLog.error({ err: err.message, sessionId }, 'Agent LiveKit connection failed (background)');
                // Retry once after a short delay
                setTimeout(() => {
                    interviewLog.info({ sessionId }, 'Retrying agent LiveKit connection...');
                    agent.start().catch(err2 => {
                        interviewLog.error({ err: err2.message, sessionId }, 'Agent LiveKit retry also failed');
                    });
                }, 2000);
            });

            // Only await the LangGraph invoke — this is what the HTTP response needs
            const resultState = await interviewAgent.invoke(initialState, config);

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
            // speakIntro plays the personalised intro (with candidate name) then the first question.
            agent.speakIntro(uniquePart).catch(err => {
                interviewLog.error({ err: err.message, sessionId }, 'Agent speakIntro error');
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
