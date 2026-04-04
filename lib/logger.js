import dotenv from 'dotenv';
dotenv.config(); // Must run before reading env vars (logger loads before env.js)

import pino from 'pino';
import pinoHttp from 'pino-http';
import { Writable } from 'stream';

// ── Grafana OTLP log shipper ──────────────────────────────────────
const OTLP_URL = process.env.GRAFANA_OTLP_URL;
const OTLP_USER = process.env.GRAFANA_OTLP_USER;
const OTLP_TOKEN = process.env.GRAFANA_OTLP_TOKEN;
const OTLP_ENABLED = !!(OTLP_URL && OTLP_USER && OTLP_TOKEN);

const PINO_LEVELS = { 10: 'TRACE', 20: 'DEBUG', 30: 'INFO', 40: 'WARN', 50: 'ERROR', 60: 'FATAL' };
let otlpBuffer = [];
let otlpTimer = null;

function flushOtlpLogs() {
    if (otlpBuffer.length === 0) return;
    const records = otlpBuffer.splice(0);
    const body = JSON.stringify({
        resourceLogs: [{
            resource: {
                attributes: [
                    { key: 'service.name', value: { stringValue: 'answerflowai-backend' } },
                    { key: 'deployment.environment', value: { stringValue: process.env.NODE_ENV || 'development' } },
                ]
            },
            scopeLogs: [{ logRecords: records }]
        }]
    });
    fetch(`${OTLP_URL}/otlp/v1/logs`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Basic ${Buffer.from(`${OTLP_USER}:${OTLP_TOKEN}`).toString('base64')}`,
        },
        body,
    }).catch(() => {});
}

function buildOtlpRecord(logObj) {
    const { level, time, msg, module, ...rest } = logObj;
    const attrs = [];
    if (module) attrs.push({ key: 'module', value: { stringValue: module } });
    if (PINO_LEVELS[level]) attrs.push({ key: 'level', value: { stringValue: PINO_LEVELS[level] } });
    for (const [k, v] of Object.entries(rest)) {
        if (k === 'pid' || k === 'hostname') continue;
        const strVal = typeof v === 'string' ? v : JSON.stringify(v);
        attrs.push({ key: k, value: { stringValue: strVal } });
    }
    return {
        timeUnixNano: String((time || Date.now()) * 1_000_000),
        severityText: PINO_LEVELS[level] || 'INFO',
        body: { stringValue: msg || '' },
        attributes: attrs,
    };
}

function queueOtlpLog(logObj) {
    otlpBuffer.push(buildOtlpRecord(logObj));
    if (otlpBuffer.length >= 50) { clearTimeout(otlpTimer); otlpTimer = null; flushOtlpLogs(); }
    else if (!otlpTimer) { otlpTimer = setTimeout(() => { otlpTimer = null; flushOtlpLogs(); }, 2000); }
}

// Custom writable stream that parses JSON log lines and ships to OTLP
const otlpStream = new Writable({
    write(chunk, _encoding, callback) {
        try {
            const parsed = JSON.parse(chunk.toString());
            queueOtlpLog(parsed);
        } catch {}
        callback();
    }
});

// ── Pino logger setup ─────────────────────────────────────────────
const usePretty = process.env.PRETTY_LOGS !== 'false';

// Build multistream: stdout (or pino-pretty) + OTLP
const streams = [];

if (usePretty) {
    // pino-pretty as a stream (not transport) so multistream works
    const pinoPretty = (await import('pino-pretty')).default;
    streams.push({ stream: pinoPretty({ colorize: true, translateTime: 'SYS:HH:MM:ss' }) });
} else {
    streams.push({ stream: process.stdout });
}

if (OTLP_ENABLED) {
    streams.push({ stream: otlpStream });
    process.on('beforeExit', flushOtlpLogs);
    process.on('SIGTERM', () => { flushOtlpLogs(); process.exit(0); });
}

const logger = pino(
    { level: process.env.LOG_LEVEL || 'info' },
    pino.multistream(streams)
);

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
export const serverLog    = logger.child({ module: 'server' });
export const authLog      = logger.child({ module: 'auth' });
export const uploadLog    = logger.child({ module: 'upload' });
export const chatLog      = logger.child({ module: 'chat' });
export const interviewLog = logger.child({ module: 'interview' });
export const agentLog     = logger.child({ module: 'agent' });
export const llmLog       = logger.child({ module: 'llm' });
export const pipelineLog  = logger.child({ module: 'pipeline' });
export const ttsLog       = logger.child({ module: 'tts' });
export const quizLog      = logger.child({ module: 'quiz' });
export const summaryLog   = logger.child({ module: 'summary' });
export const embeddingLog = logger.child({ module: 'embedding' });
export const routerLog   = logger.child({ module: 'router' });
