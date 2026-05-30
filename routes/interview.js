import { Router } from 'express';
import { registerVectorStore, clearSessionInterviewState, parseTTSResponse } from "../lib/interview/interviewAgent.js";
import { InterviewAgentWorker } from "../agents/interview/worker.js";
import { optionalAuth } from "../middleware/auth.js";
import { getDocumentBySessionId, saveInterviewResult, logActivity, upsertActiveInterview, getActiveInterview, clearActiveInterview } from "../lib/db.js";
import { ensureSession } from "../lib/sessionRestore.js";
import { updateUserProfileAfterInterview, getUserProfileContext } from "../lib/interview/profileUpdater.js";
import { generateGreeting, generateSimpleGreeting } from "../lib/interview/greetingGenerator.js";
import { validate, interviewStartSchema } from '../lib/validation.js';
import { interviewLog } from '../lib/logger.js';
import { cleanupSessionAgents } from './conversationalAI.js';

export function createInterviewRoutes({ sessionCache, activeAgents, activeVoiceAgents, clientReadyResolvers, io, interviewAgent }) {
    const router = Router();

    router.post('/interview/start', validate(interviewStartSchema), optionalAuth, async (req, res) => {
        try {
            const { sessionId, maxQuestions = 5, timezone } = req.validated;

            // Try in-memory first, then auto-restore from DB+Qdrant
            const session = await ensureSession(sessionCache, sessionId);
            if (!session || !session.vectorStore) {
                return res.status(404).json({ error: 'Please upload a PDF or text document first.' });
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

            // Generate personalized greeting based on timezone, mood, difficulty
            let greetingData;
            try {
                greetingData = generateGreeting(
                    req.user?.name || "there",
                    timezone || 'UTC',
                    userProfileContext,
                    "medium"
                );
            } catch (err) {
                interviewLog.warn({ err: err.message }, 'Greeting generation failed, using fallback');
                greetingData = generateSimpleGreeting(req.user?.name || "there", "medium");
            }

            const initialState = {
                sessionId,
                maxQuestions: parseInt(maxQuestions),
                difficultyLevel: "medium",
                chatHistory: [],
                questionsAsked: 0,
                topicsUsed: [],
                userProfileContext,
                candidateName: req.user?.name || "there",
                timeGreeting: greetingData.timeGreeting,
            };

            // ── RESUME CHECK ─────────────────────────────────────────
            // Look up any in-progress interview for this session in DB.
            // If found, reconnect to its LangGraph checkpoint via the
            // stored thread_id instead of starting fresh.
            let isResume = false;
            let interviewRunId = `${sessionId}_${Date.now()}`;

            const activeRecord = await getActiveInterview(sessionId).catch(() => null);
            if (activeRecord?.thread_id) {
                // Verify the checkpoint actually exists by reading the state
                try {
                    const existingConfig = { configurable: { thread_id: activeRecord.thread_id } };
                    const snap = await interviewAgent.getState(existingConfig);
                    // snap.values exists and interview wasn't finished
                    if (snap?.values && !snap.values.finalReport) {
                        interviewRunId = activeRecord.thread_id;
                        isResume = true;
                        interviewLog.info({
                            sessionId,
                            threadId: interviewRunId,
                            questionsAsked: snap.values.questionsAsked,
                            currentQuestion: snap.values.currentQuestion?.substring(0, 80),
                        }, 'Resuming existing interview from checkpoint');
                    }
                } catch (snapErr) {
                    interviewLog.warn({ sessionId, err: snapErr.message }, 'Checkpoint read failed — starting fresh');
                }
            }

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
                // Clear the active interview record — interview is complete
                clearActiveInterview(sessionId).catch(() => {});
                if (req.user) {
                    try {
                        const doc = await getDocumentBySessionId(sessionId);
                        if (doc) {
                            await saveInterviewResult({
                                userId: req.user.id,
                                documentId: doc.id,
                                threadId: interviewRunId,
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

            // Parallelize agent.start() and LangGraph invoke
            const agentStartPromise = agent.start().catch(err => {
                interviewLog.error({ err: err.message, sessionId }, 'Agent LiveKit connection failed');
                setTimeout(() => {
                    agent.start().catch(err2 => {
                        interviewLog.error({ err: err2.message, sessionId }, 'Agent LiveKit retry failed');
                    });
                }, 2000);
            });

            let resultState;
            if (isResume) {
                // Load existing state from checkpoint — no re-invoke needed
                const snap = await interviewAgent.getState(config);
                resultState = snap.values;
            } else {
                resultState = await interviewAgent.invoke(initialState, config);
                // Save the new thread_id so it can be resumed on reload
                upsertActiveInterview({
                    sessionId,
                    userId: req.user?.id || null,
                    threadId: interviewRunId,
                    maxQuestions: parseInt(maxQuestions),
                    questionsAsked: resultState.questionsAsked || 0,
                    currentQuestion: resultState.currentQuestion || '',
                }).catch(() => {});
            }

            session.interviewStateConfig = config;

            if (req.user) {
                logActivity({
                    userId: req.user.id,
                    action: isResume ? 'interview_resumed' : 'interview_started',
                    metadata: { sessionId, maxQuestions: parseInt(maxQuestions) }
                });
            }

            const { uniquePart } = parseTTSResponse(resultState.currentQuestion);

            // Ensure agent connected before speaking
            interviewLog.info({ sessionId, isResume }, 'Waiting for agent ready + client audio ready');
            await Promise.all([
                Promise.race([agentStartPromise, new Promise(r => setTimeout(r, 3000))]),
                new Promise((resolve) => {
                    clientReadyResolvers.set(sessionId, resolve);
                    setTimeout(() => {
                        if (clientReadyResolvers.has(sessionId)) {
                            interviewLog.warn({ sessionId }, 'client_audio_ready timeout');
                            clientReadyResolvers.delete(sessionId);
                        }
                        resolve();
                    }, 2000);
                })
            ]);

            interviewLog.info({ sessionId, isResume }, 'Speaking intro/resume');
            agent.speakIntro(uniquePart, isResume).catch(err => {
                interviewLog.error({ err: err.message, sessionId }, 'speakIntro error');
            });

            res.json({
                questionNumber: resultState.questionsAsked,
                difficulty: resultState.difficultyLevel,
                agentStarted: true,
                resumed: isResume,
            });

        } catch (error) {
            interviewLog.error({ err: error.message }, 'Interview start error');
            res.status(500).json({ error: error.message || "Failed to start interview" });
        }
    });

    return router;
}
