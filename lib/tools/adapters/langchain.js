import { tool } from '@langchain/core/tools';
import { selectTools, runTool } from '../registry.js';

/**
 * Build LangChain tool objects for use with `bindTools()`, `createReactAgent()`
 * or a LangGraph `ToolNode`.
 *
 * @param {string[]} [names] - tool names to expose; omit for all
 * @param {object|Function} ctx - injected tool context, or a function returning
 *   one (see toLiveKit for why).
 * @returns {object[]} LangChain tools
 *
 * @example
 * const model = llm.bindTools(toLangChain(['search_pdf'], { sessionId, session }));
 */
export function toLangChain(names, ctx) {
    const resolveCtx = typeof ctx === 'function' ? ctx : () => ctx;

    return selectTools(names).map(([name, def]) =>
        tool(async args => runTool(name, args, { ...resolveCtx(), logType: 'langchain' }), {
            name,
            description: def.description,
            schema: def.schema,
        })
    );
}
