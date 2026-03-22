import pino from 'pino';

const logger = pino({
    level: process.env.LOG_LEVEL || 'info',
    transport: process.env.NODE_ENV !== 'production'
        ? { target: 'pino-pretty', options: { colorize: true, translateTime: 'SYS:HH:MM:ss' } }
        : undefined
});

export default logger;

// Convenience child loggers for each module
export const serverLog   = logger.child({ module: 'server' });
export const authLog     = logger.child({ module: 'auth' });
export const uploadLog   = logger.child({ module: 'upload' });
export const chatLog     = logger.child({ module: 'chat' });
export const interviewLog = logger.child({ module: 'interview' });
export const agentLog    = logger.child({ module: 'agent' });
export const llmLog      = logger.child({ module: 'llm' });
export const pipelineLog = logger.child({ module: 'pipeline' });
export const ttsLog      = logger.child({ module: 'tts' });
