import { llm } from '@livekit/agents';
import { selectTools, runTool } from '../registry.js';

/**
 * Build the `tools` map for LiveKitAgentSession / the LiveKit voice pipeline.
 *
 * @param {string[]} [names] - tool names to expose; omit for all
 * @param {object|Function} ctx - injected tool context, or a function returning
 *   one. Pass a function when the context is resolved per call (e.g. the
 *   session entry is only populated after upload completes).
 * @returns {Object<string, object>} map of tool name -> LiveKit tool
 *
 * @example
 * const agent = new LiveKitAgentSession({
 *   sessionId,
 *   tools: toLiveKit(['search_pdf'], () => ({ sessionId, session: sessionCache[sessionId], io })),
 * });
 */
export function toLiveKit(names, ctx) {
    const resolveCtx = typeof ctx === 'function' ? ctx : () => ctx;

    return Object.fromEntries(
        selectTools(names).map(([name, def]) => [
            name,
            llm.tool({
                description: def.description,
                parameters: def.schema,
                execute: async args => runTool(name, args, { ...resolveCtx(), logType: 'livekit' }),
            }),
        ])
    );
}
