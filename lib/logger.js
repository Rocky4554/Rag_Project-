import pino from 'pino';
import pinoHttp from 'pino-http';

// Use pretty printing unless explicitly disabled (e.g. PRETTY_LOGS=false in production)
const usePretty = process.env.PRETTY_LOGS !== 'false';

const logger = pino({
    level: process.env.LOG_LEVEL || 'info',
    transport: usePretty
        ? { target: 'pino-pretty', options: { colorize: true, translateTime: 'SYS:HH:MM:ss' } }
        : undefined
});

export default logger;

// HTTP request logger middleware
export const httpLogger = pinoHttp({
    logger,
    autoLogging: true,
    customLogLevel(req, res, err) {
        if (res.statusCode >= 500 || err) return 'error';
        if (res.statusCode >= 400) return 'warn';
        return 'info';
    },
    customSuccessMessage(req, res, responseTime) {
        return `${req.method} ${req.url} ${res.statusCode} ${Math.round(responseTime)}ms`;
    },
    customErrorMessage(req, res, err) {
        return `${req.method} ${req.url} ${res.statusCode} - ${err.message}`;
    },
    serializers: {
        req(req) {
            return {
                method: req.method,
                url: req.url,
                ...(req.headers['user-agent'] && { ua: req.headers['user-agent'] }),
            };
        },
        res(res) {
            return { statusCode: res.statusCode };
        },
    },
});

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
