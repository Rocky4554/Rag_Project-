import { GoogleGenAI, Modality } from '@google/genai';
import {
    Room,
    RoomEvent,
    TrackKind,
    AudioStream,
    LocalAudioTrack,
    TrackSource,
} from '@livekit/rtc-node';
import { AccessToken } from 'livekit-server-sdk';
import { AudioPublisher } from '../shared/audioPublisher.js';
import { getUserProfileContext } from '../../lib/interview/profileUpdater.js';
import { agentLog } from '../../lib/logger.js';

// Override via GEMINI_LIVE_MODEL env var. Run: node scripts/listLiveModels.js to see options.
const VOICE_MODEL = process.env.GEMINI_LIVE_MODEL || 'gemini-2.5-flash-native-audio-latest';
const LIVE_API_VERSION = process.env.GEMINI_LIVE_API_VERSION || 'v1beta';
const INPUT_SAMPLE_RATE = 16000;   // LiveKit mic → Gemini
const OUTPUT_SAMPLE_RATE = 24000;  // Gemini audio response → LiveKit

export class VoiceAgentWorker {
    constructor(sessionId, sessionCache, io, userId = null, userName = null) {
        this.sessionId = sessionId;
        this.sessionCache = sessionCache;
        this.io = io;
        this.userId = userId;
        this.userName = userName;

        this.room = new Room();
        this.audioPublisher = new AudioPublisher(OUTPUT_SAMPLE_RATE, 1);
        this.geminiSession = null;
        this.isActive = false;

        // Progressive audio playback queue
        this._audioQueue = [];
        this._playingAudio = false;
    }

    async start() {
        const startTs = performance.now();
        agentLog.info({ sessionId: this.sessionId }, 'VoiceAgent starting');

        const aiOptions = { apiKey: process.env.GEMINI_API_KEY };
        if (LIVE_API_VERSION !== 'v1beta') {
            aiOptions.httpOptions = { apiVersion: LIVE_API_VERSION };
        }
        const ai = new GoogleGenAI(aiOptions);

        // 1. Build system prompt with user context and PDF summary
        const promptTs = performance.now();
        agentLog.info({ sessionId: this.sessionId, type: 'voice' }, '[1/4] Building system prompt...');
        const systemPrompt = await this._buildSystemPrompt();
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - promptTs), chars: systemPrompt.length, type: 'voice' }, '[1/4] System prompt ready');

        // 2. Open Gemini Live WebSocket
        const geminiTs = performance.now();
        agentLog.info({ sessionId: this.sessionId, model: VOICE_MODEL, apiVersion: LIVE_API_VERSION, type: 'voice' }, '[2/4] Connecting to Gemini Live...');
        const self = this;
        this.geminiSession = await ai.live.connect({
            model: VOICE_MODEL,
            config: {
                responseModalities: [Modality.AUDIO],
                systemInstruction: { parts: [{ text: systemPrompt }] },
                tools: [
                    {
                        functionDeclarations: [{
                            name: 'search_pdf',
                            description: "Search the user's uploaded PDF document for specific information",
                            parameters: {
                                type: 'OBJECT',
                                properties: {
                                    query: { type: 'STRING', description: 'Search query to find relevant content in the PDF' }
                                },
                                required: ['query']
                            }
                        }]
                    },
                    { googleSearch: {} }
                ],
                inputAudioTranscription: {},
                outputAudioTranscription: {},
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: 'Puck' }
                    }
                }
            },
            callbacks: {
                onopen: () => {
                    agentLog.info({ sessionId: self.sessionId, ms: Math.round(performance.now() - geminiTs), model: VOICE_MODEL, type: 'voice' }, '[2/4] Gemini Live connected');
                },
                onmessage: (msg) => {
                    self._handleGeminiMessage(msg);
                },
                onerror: (e) => {
                    agentLog.error({ sessionId: self.sessionId, err: e?.message || String(e), type: 'voice' }, 'Gemini Live error');
                },
                onclose: (e) => {
                    const reason = e?.reason || '';
                    agentLog.info({ sessionId: self.sessionId, reason, type: 'voice' }, 'Gemini Live closed');
                    if (self.isActive) {
                        self.isActive = false;
                        self.stop();
                    } else if (reason) {
                        // Closed before we finished starting — propagate the error
                        if (self.io) {
                            self.io.to(self.sessionId).emit('voice_error', { error: reason });
                        }
                    }
                }
            }
        });

        // 3. Connect to LiveKit room
        const livekitTs = performance.now();
        agentLog.info({ sessionId: this.sessionId, type: 'voice' }, '[3/4] Connecting to LiveKit room...');
        const token = await this._generateToken();
        this._setupRoomEvents();
        await this.room.connect(process.env.LIVEKIT_URL, token);
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - livekitTs), type: 'voice' }, '[3/4] LiveKit connected');

        // 4. Publish AI audio output track
        const trackTs = performance.now();
        agentLog.info({ sessionId: this.sessionId, type: 'voice' }, '[4/4] Publishing audio track...');
        const track = LocalAudioTrack.createAudioTrack('voice-agent-audio', this.audioPublisher.source);
        await this.room.localParticipant.publishTrack(track, {
            name: 'voice-agent-audio',
            source: TrackSource.SOURCE_MICROPHONE
        });

        this.isActive = true;
        agentLog.info({ sessionId: this.sessionId, ms: Math.round(performance.now() - trackTs), totalMs: Math.round(performance.now() - startTs), type: 'voice' }, '[4/4] VoiceAgent ready');

        // 5. Trigger opening greeting — small delay lets the LiveKit track settle
        setTimeout(() => {
            if (this.isActive && this.geminiSession) {
                const greetMsg = this.userName
                    ? `The user just joined. Greet them now — say "Hello ${this.userName}!" and briefly mention you're ready to help.`
                    : `The user just joined. Greet them now and briefly mention you're ready to help.`;
                this.geminiSession.sendClientContent({
                    turns: [{ role: 'user', parts: [{ text: greetMsg }] }],
                    turnComplete: true
                });
            }
        }, 1000);
    }

    async _buildSystemPrompt() {
        const session = this.sessionCache[this.sessionId];
        const name = this.userName || 'there';
        const docName = session?.originalName || null;

        // Get a snippet of the document content for context
        let docContext = '';
        if (session?.vectorStore) {
            try {
                const docs = await session.vectorStore.similaritySearch('introduction summary overview', 5);
                docContext = docs.map(d => d.pageContent).join('\n\n').substring(0, 1500);
            } catch (err) {
                agentLog.warn({ err: err.message, type: 'voice' }, 'Failed to fetch doc context for voice agent');
            }
        }

        // Get long-term user profile if authenticated
        let profileContext = '';
        
        if (this.userId) {
            try {
                profileContext = (await getUserProfileContext(this.userId)) || '';
            } catch (err) {
                agentLog.warn({ err: err.message, type: 'voice' }, 'Failed to fetch user profile for voice agent');
            }
        }

        let prompt = `You are a helpful, friendly AI voice assistant. Speak naturally, clearly, and conversationally.`;

        if (name !== 'there') {
            prompt += ` The user's name is ${name}. Greet them by name when the conversation starts.`;
        }

        if (docName) {
            prompt += `\n\nThe user has uploaded a document called "${docName}". Use the search_pdf tool whenever you need to look up specific information from this document. You also have Google Search available for general knowledge questions.`;
        } else {
            prompt += `\n\nNo document is currently uploaded. Use Google Search for factual questions and answer from your own knowledge otherwise.`;
        }

        if (docContext) {
            prompt += `\n\nDocument preview (first few sections to give you context):\n${docContext}`;
        }

        if (profileContext) {
            prompt += `\n\nUser background (from previous sessions):\n${profileContext}`;
        }

        prompt += `\n\nBehavior guidelines:
- When you receive an internal system message like "The user just joined. Greet them now", respond with a warm spoken greeting immediately — do not wait for the user to speak first.
- Keep responses concise and conversational — 1 to 3 sentences is usually ideal for voice
- When searching the PDF, summarize the findings naturally in speech rather than reading verbatim
- Don't read out long lists — summarize the key points instead
- If asked something you don't know, say so honestly and offer to search the document or web
- You can have casual conversation, answer general questions, or help the user understand their document`;

        return prompt;
    }

    async _generateToken() {
        const at = new AccessToken(process.env.LIVEKIT_API_KEY, process.env.LIVEKIT_API_SECRET, {
            identity: 'voice-agent',
            name: 'Voice Assistant'
        });
        at.addGrant({
            roomJoin: true,
            room: this.sessionId,
            canPublish: true,
            canSubscribe: true
        });
        return await at.toJwt();
    }

    _setupRoomEvents() {
        this.room.on(RoomEvent.TrackSubscribed, (track, _publication, participant) => {
            if (track.kind === TrackKind.KIND_AUDIO) {
                agentLog.info({ sessionId: this.sessionId, participant: participant.identity, type: 'voice' }, 'Subscribed to user audio');
                this._listenToUserAudio(track);
            }
        });

        this.room.on(RoomEvent.Disconnected, () => {
            agentLog.info({ sessionId: this.sessionId, type: 'voice' }, 'LiveKit disconnected');
            this.stop();
        });
    }

    _listenToUserAudio(track) {
        // Pass INPUT_SAMPLE_RATE so LiveKit resamples browser audio (48kHz) → 16kHz for us.
        // Without this, Gemini receives 48kHz PCM labeled as 16kHz and hears garbage.
        const audioStream = new AudioStream(track, INPUT_SAMPLE_RATE, 1);
        (async () => {
            try {
                let loggedOnce = false;
                for await (const frame of audioStream) {
                    if (!this.isActive || !this.geminiSession) break;
                    if (!frame.data || frame.data.length === 0) continue;

                    // Log actual sample rate on first frame to confirm resampling worked
                    if (!loggedOnce) {
                        agentLog.info(
                            { sessionId: this.sessionId, actualRate: frame.sampleRate, channels: frame.channels, samplesPerChannel: frame.samplesPerChannel, type: 'voice' },
                            'First audio frame received'
                        );
                        loggedOnce = true;
                    }

                    const buf = Buffer.from(frame.data.buffer, frame.data.byteOffset, frame.data.byteLength);
                    this.geminiSession.sendRealtimeInput({
                        audio: { data: buf.toString('base64'), mimeType: `audio/pcm;rate=${INPUT_SAMPLE_RATE}` }
                    });
                }
                agentLog.debug({ sessionId: this.sessionId, type: 'voice' }, 'User audio stream ended');
            } catch (err) {
                agentLog.error({ sessionId: this.sessionId, err: err.message, type: 'voice' }, 'User audio stream error');
            }
        })();
    }

    _handleGeminiMessage(msg) {
        // Log top-level keys at debug level to help diagnose message structure
        agentLog.debug(
            { keys: Object.keys(msg).join(','), hasServerContent: !!msg.serverContent, type: 'voice' },
            'Gemini message received'
        );

        // Tool call request from Gemini
        if (msg.toolCall?.functionCalls?.length > 0) {
            this._handleToolCall(msg.toolCall.functionCalls).catch(err => {
                agentLog.error({ sessionId: this.sessionId, err: err.message, type: 'voice' }, 'Tool call error');
            });
        }

        if (msg.serverContent) {
            // Log serverContent keys at debug level
            agentLog.debug(
                { scKeys: Object.keys(msg.serverContent).join(','), type: 'voice' },
                'serverContent keys'
            );

            // Barge-in: user started speaking, interrupt playback
            if (msg.serverContent.interrupted) {
                agentLog.debug({ sessionId: this.sessionId, type: 'voice' }, 'Interrupted by user barge-in');
                this._stopSpeaking();
                if (this.io) {
                    this.io.to(this.sessionId).emit('voice_state', { state: 'listening' });
                }
            }

            // Audio output + optional inline text from model turn parts
            if (msg.serverContent.modelTurn?.parts) {
                let turnText = '';
                for (const part of msg.serverContent.modelTurn.parts) {
                    if (part.inlineData?.mimeType?.startsWith('audio/')) {
                        const buf = Buffer.from(part.inlineData.data, 'base64');
                        const pcm = new Int16Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
                        this._audioQueue.push(pcm);
                        this._processAudioQueue();
                        if (this.io) {
                            this.io.to(this.sessionId).emit('voice_state', { state: 'speaking' });
                        }
                    }
                    // Some models return text alongside audio
                    if (part.text) {
                        turnText += part.text;
                    }
                }
                if (turnText && this.io) {
                    this.io.to(this.sessionId).emit('voice_transcript', { role: 'ai', text: turnText });
                }
            }

            // Turn complete — signal AI is done speaking
            if (msg.serverContent.turnComplete) {
                agentLog.debug({ sessionId: this.sessionId, type: 'voice' }, 'Gemini turn complete');
                if (this.io) {
                    this.io.to(this.sessionId).emit('voice_state', { state: 'listening' });
                }
            }

            // Input transcription (what the user said) — field name varies by model
            const inputText = msg.serverContent.inputTranscription?.text
                ?? msg.serverContent.inputTranscription?.transcript;
            if (inputText) {
                agentLog.info({ sessionId: this.sessionId, text: inputText.substring(0, 150), words: inputText.trim().split(/\s+/).length, type: 'voice' }, '🎤 User said');
                if (this.io) {
                    this.io.to(this.sessionId).emit('voice_transcript', { role: 'user', text: inputText });
                }
            }

            // Output transcription (what the AI said) — field name varies by model
            const outputText = msg.serverContent.outputTranscription?.text
                ?? msg.serverContent.outputTranscription?.transcript;
            if (outputText) {
                agentLog.info({ sessionId: this.sessionId, text: outputText.substring(0, 150), type: 'voice' }, '🤖 AI said');
                if (this.io) {
                    this.io.to(this.sessionId).emit('voice_transcript', { role: 'ai', text: outputText });
                }
            }
        }
    }

    async _handleToolCall(functionCalls) {
        const session = this.sessionCache[this.sessionId];
        const responses = [];

        for (const fn of functionCalls) {
            if (fn.name === 'search_pdf') {
                agentLog.info({ sessionId: this.sessionId, query: fn.args?.query, type: 'voice' }, 'search_pdf tool called');
                try {
                    if (!session?.vectorStore) {
                        responses.push({ id: fn.id, name: fn.name, response: { output: 'No document is currently loaded.' } });
                    } else {
                        const docs = await session.vectorStore.similaritySearch(fn.args.query, 3);
                        const result = docs.map(d => d.pageContent).join('\n\n') || 'No relevant content found.';
                        agentLog.info({ sessionId: this.sessionId, resultChars: result.length, preview: result.substring(0, 80), type: 'voice' }, 'search_pdf result');
                        responses.push({ id: fn.id, name: fn.name, response: { output: result } });
                    }
                } catch (err) {
                    responses.push({ id: fn.id, name: fn.name, response: { output: 'Failed to search the document.' } });
                }
            }
        }

        if (responses.length > 0 && this.geminiSession) {
            this.geminiSession.sendToolResponse({ functionResponses: responses });
        }
    }

    async _processAudioQueue() {
        if (this._playingAudio) return;
        this._playingAudio = true;
        try {
            while (this._audioQueue.length > 0) {
                const chunk = this._audioQueue.shift();
                if (chunk.length > 0) {
                    await this.audioPublisher.pushPCM(chunk);
                }
            }
        } catch (err) {
            agentLog.error({ sessionId: this.sessionId, err: err.message, type: 'voice' }, 'Audio queue playback error');
        } finally {
            this._playingAudio = false;
        }
    }

    _stopSpeaking() {
        this._audioQueue = [];
        this.audioPublisher.stop();
    }

    stop() {
        if (!this.isActive) return;
        agentLog.info({ sessionId: this.sessionId, type: 'voice' }, 'VoiceAgent stopping');
        this.isActive = false;
        this._stopSpeaking();
        try { this.geminiSession?.close(); } catch (_) {}
        try { this.room.disconnect(); } catch (_) {}
    }
}
