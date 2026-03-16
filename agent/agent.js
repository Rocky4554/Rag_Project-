import { Room, LocalAudioTrack, RemoteAudioTrack, RoomEvent } from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import dotenv from "dotenv";

import { DeepgramSTT } from "./stt.js";
import { AudioPublisher } from "./audioPublisher.js";
import { generatePCM } from "./tts.js";
import { SessionBridge } from "./sessionBridge.js";
import { parseTTSResponse } from "../lib/interviewAgent.js";

dotenv.config();

export class InterviewAgentWorker {
    /**
     * @param {string} sessionId - Unique room name / interview session
     * @param {Object} sessionCache - Ref to the server.js global sessionCache
     * @param {Object} agentWorkflow - Ref to the compiled LangGraph workflow
     * @param {Object} io - Socket.io instance for UI updates (optional)
     */
    constructor(sessionId, sessionCache, agentWorkflow, io) {
        this.sessionId = sessionId;
        this.sessionBridge = new SessionBridge(sessionId, sessionCache, agentWorkflow);
        this.io = io;
        
        this.room = new Room();
        this.stt = new DeepgramSTT();
        // Polly PCM is most reliable at 16kHz; publish the same rate to LiveKit source.
        this.audioPublisher = new AudioPublisher(16000, 1);
        
        this.isActive = false;
        this.processingTurn = false;
    }

    async start() {
        console.log(`[AgentWorker] Starting for session: ${this.sessionId}`);

        // 1. Generate token for the AI agent (toJwt() is async in newer livekit-server-sdk)
        const token = await this.generateToken();

        // 2. Connect STT
        this.stt.start();

        // 3. Connect to LiveKit Room
        this.setupRoomEvents();
        await this.room.connect(process.env.LIVEKIT_URL, token);
        console.log(`[AgentWorker] Connected to room: ${this.sessionId}`);

        // 4. Publish AI voice track
        const track = LocalAudioTrack.createAudioTrack("ai-interviewer-audio", this.audioPublisher.source);
        await this.room.localParticipant.publishTrack(track, {
            name: "ai-interviewer-audio",
            source: 1 // TrackSource.SOURCE_MICROPHONE
        });
        console.log(`[AgentWorker] Published AI audio track`);

        this.isActive = true;
    }

    async generateToken() {
        const at = new AccessToken(process.env.LIVEKIT_API_KEY, process.env.LIVEKIT_API_SECRET, {
            identity: "ai-interviewer",
            name: "AI Interviewer",
        });
        at.addGrant({
            roomJoin: true,
            room: this.sessionId,
            canPublish: true,
            canSubscribe: true,
        });
        return await at.toJwt();
    }

    setupRoomEvents() {
        this.room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
            if (track.kind === "audio" && track instanceof RemoteAudioTrack) {
                console.log(`[AgentWorker] Subscribed to audio from: ${participant.identity}`);
                this.listenToUserAudio(track);
            }
        });

        this.room.on(RoomEvent.Disconnected, () => {
            console.log(`[AgentWorker] Disconnected from room.`);
            this.stop();
        });

        // Handle full transcript from STT
        this.stt.on("transcript", async (text) => {
            if (!this.isActive || this.processingTurn) return;
            
            // Basic VAD: don't process if the AI is currently speaking
            if (this.audioPublisher.isSpeaking) {
                console.log(`[AgentWorker] Ignored transcript (AI is speaking)`);
                return;
            }

            await this.handleUserTurn(text);
        });
    }

    listenToUserAudio(track) {
        // Create an audio stream listener from the candidate's remote track
        // The track must be attached to an AudioStream
        // Note: in @livekit/rtc-node, we use track.receiver or a stream
        // According to current rtc-node docs:
        track.on("audioData", (audioData) => {
            // Only send audio to STT if we are not actively processing a turn
            if (!this.processingTurn && !this.audioPublisher.isSpeaking) {
                // audioData is { data: Int16Array, sampleRate: number, numChannels: number }
                if (audioData.data) {
                    this.stt.pushAudio(audioData.data);
                }
            }
        });
    }

    async handleUserTurn(transcript) {
        this.processingTurn = true;
        
        // Notify UI that we heard something
        if (this.io) {
            this.io.to(this.sessionId).emit('ai_state', { state: 'thinking', text: 'Evaluating your answer...' });
            this.io.to(this.sessionId).emit('transcript_final', { role: 'user', text: transcript });
        }

        try {
            // 1. Process via LangGraph
            const result = await this.sessionBridge.processUserTranscript(transcript);
            
            // 2. Notify UI of feedback/score
            if (this.io && result.evaluation) {
                this.io.to(this.sessionId).emit('ai_feedback', {
                    feedback: result.evaluation.feedback,
                    score: result.evaluation.score,
                    answerQuality: result.answerQuality
                });
            }

            if (result.done) {
                // Interview over
                if (this.io) {
                    this.io.to(this.sessionId).emit('interview_done', { report: result.finalReport });
                }
                
                // Play outro (you could optionally TTS a custom goodbye here)
                const pcm = await generatePCM("Thank you for your time. The interview is now complete. You can review your report on the screen.");
                if (pcm) await this.audioPublisher.pushPCM(pcm);
                
                setTimeout(() => this.stop(), 2000);
            } else {
                // Next question
                if (this.io) {
                    this.io.to(this.sessionId).emit('transcript_final', { role: 'ai', text: result.nextQuestion });
                    this.io.to(this.sessionId).emit('ai_state', { state: 'speaking', text: 'AI Speaking...' });
                }

                // Strip TTS tags before speaking
                const { uniquePart: cleanFeedback } = parseTTSResponse(result.evaluation.feedback || "");
                const { uniquePart: cleanQuestion } = parseTTSResponse(result.nextQuestion || "");
                
                const spokenText = `${cleanFeedback} ${cleanQuestion}`.trim();
                
                if (spokenText) {
                    const pcmData = await generatePCM(spokenText, "16000");
                    if (pcmData) {
                        await this.audioPublisher.pushPCM(pcmData);
                    }
                }

                if (this.io) {
                    this.io.to(this.sessionId).emit('ai_state', { state: 'listening', text: 'Listening...' });
                }
            }

        } catch (err) {
            console.error(`[AgentWorker] Turn error:`, err);
        } finally {
            this.processingTurn = false;
        }
    }

    /**
     * Can be called manually to make the agent speak (e.g. for the very first question)
     */
    async speak(text) {
        if (!text) return;
        this.processingTurn = true;
        try {
            const { uniquePart } = parseTTSResponse(text);
            if (uniquePart) {
                const pcm = await generatePCM(uniquePart, "16000");
                if (pcm) await this.audioPublisher.pushPCM(pcm);
            }
        } finally {
            this.processingTurn = false;
        }
    }

    stop() {
        console.log(`[AgentWorker] Stopping session: ${this.sessionId}`);
        this.isActive = false;
        this.stt.stop();
        this.audioPublisher.stop();
        this.room.disconnect();
    }
}