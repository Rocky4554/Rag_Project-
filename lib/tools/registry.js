import { z } from 'zod';
import { agentLog } from '../logger.js';

/**
 * Canonical tool registry.
 *
 * One definition per tool, consumed by every agent framework through the
 * adapters in ./adapters/. Add a tool here and it becomes available to the
 * Gemini Live voice worker, LiveKit pipeline agents, and LangChain/LangGraph
 * agents at once.
 *
 * Each tool is:
 *   description  - shown to the model
 *   schema       - zod object describing the model-supplied arguments
 *   execute      - async (args, ctx) => string
 *
 * `ctx` is injected by the adapter and is never model-supplied. It carries:
 *   sessionId    - current session / LiveKit room name
 *   session      - the sessionCache entry (vectorStore, originalName, ...)
 *   io           - Socket.io instance, may be null
 *   endSession   - optional callback used by end_session to stop the agent
 *
 * execute() must return a string — that is what gets handed back to the model.
 * Throwing is allowed; adapters catch and substitute the tool's error message.
 */
export const tools = {
    search_pdf: {
        description: "Search the user's uploaded PDF document for specific information",
        schema: z.object({
            query: z.string().describe('Search query to find relevant content in the PDF'),
        }),
        errorMessage: 'Failed to search the document.',
        execute: async ({ query }, ctx) => {
            if (!ctx.session?.vectorStore) return 'No document is currently loaded.';
            const docs = await ctx.session.vectorStore.similaritySearch(query, 3);
            return docs.map(d => d.pageContent).join('\n\n') || 'No relevant content found.';
        },
    },

    end_session: {
        description:
            'End the current conversation session and disconnect gracefully when requested by the user.',
        schema: z.object({}),
        errorMessage: 'Failed to end the session.',
        execute: async (_args, ctx) => {
            // The transport decides how to actually tear down; the tool only
            // signals intent so the model gets a spoken confirmation first.
            ctx.endSession?.();
            return 'Disconnecting now. Goodbye!';
        },
    },

    web_search: {
        description:
            'Search the public web for current information not found in the uploaded document (news, facts after the model\'s knowledge cutoff, general knowledge).',
        schema: z.object({
            query: z.string().describe('Search query'),
        }),
        errorMessage: 'Web search is unavailable right now.',
        execute: async ({ query }) => {
            const apiKey = process.env.TAVILY_API_KEY;
            if (!apiKey) return 'Web search is not configured.';

            const res = await fetch('https://api.tavily.com/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    api_key: apiKey,
                    query,
                    max_results: 5,
                    include_answer: true,
                }),
            });
            if (!res.ok) throw new Error(`Tavily returned ${res.status}`);
            const data = await res.json();

            const parts = [];
            if (data.answer) parts.push(data.answer);
            for (const r of data.results || []) {
                parts.push(`${r.title}: ${r.content}\n(${r.url})`);
            }
            return parts.join('\n\n') || 'No results found.';
        },
    },
};

/**
 * Look up tool definitions by name, failing loudly on typos so a missing tool
 * surfaces at agent startup rather than as a silent no-op mid-conversation.
 *
 * @param {string[]} [names] - tool names; omit to select every registered tool
 * @returns {Array<[string, object]>} [name, definition] pairs
 */
export function selectTools(names) {
    const selected = names ?? Object.keys(tools);
    return selected.map(name => {
        const def = tools[name];
        if (!def) {
            throw new Error(
                `Unknown tool "${name}". Registered tools: ${Object.keys(tools).join(', ')}`
            );
        }
        return [name, def];
    });
}

/**
 * Run a tool by name with centralised validation, logging and error handling,
 * so every adapter behaves identically. Always resolves to a string.
 *
 * @param {string} name - registered tool name
 * @param {object} args - raw, model-supplied arguments (validated here)
 * @param {object} ctx  - injected context (see module docs)
 * @returns {Promise<string>}
 */
export async function runTool(name, args, ctx) {
    const [[, def]] = selectTools([name]);
    const log = { sessionId: ctx?.sessionId, tool: name, type: ctx?.logType || 'tool' };

    const parsed = def.schema.safeParse(args ?? {});
    if (!parsed.success) {
        agentLog.warn({ ...log, issues: parsed.error.issues }, 'Tool called with invalid arguments');
        return `Invalid arguments for ${name}: ${parsed.error.issues.map(i => i.message).join('; ')}`;
    }

    agentLog.info({ ...log, args: parsed.data }, 'Tool called');
    const startTs = performance.now();
    try {
        const result = await def.execute(parsed.data, ctx ?? {});
        const output = typeof result === 'string' ? result : JSON.stringify(result);
        agentLog.info(
            {
                ...log,
                ms: Math.round(performance.now() - startTs),
                resultChars: output.length,
                preview: output.substring(0, 80),
            },
            'Tool result'
        );
        return output;
    } catch (err) {
        agentLog.error(
            { ...log, ms: Math.round(performance.now() - startTs), err: err.message },
            'Tool failed'
        );
        return def.errorMessage || `The ${name} tool failed.`;
    }
}
