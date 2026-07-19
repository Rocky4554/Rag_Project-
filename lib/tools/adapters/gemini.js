import { z } from 'zod';
import { selectTools, runTool } from '../registry.js';

const JSON_TYPE_TO_GEMINI = {
    object: 'OBJECT',
    string: 'STRING',
    number: 'NUMBER',
    integer: 'INTEGER',
    boolean: 'BOOLEAN',
    array: 'ARRAY',
};

/**
 * Convert a JSON Schema node into the uppercase-Type dialect the Gemini Live
 * API expects, dropping the keywords it rejects ($schema, additionalProperties).
 */
function toGeminiSchema(node) {
    if (!node || typeof node !== 'object') return node;

    const out = {};
    if (node.type) {
        const mapped = JSON_TYPE_TO_GEMINI[node.type];
        if (!mapped) throw new Error(`Unsupported JSON Schema type for Gemini: ${node.type}`);
        out.type = mapped;
    }
    if (node.description) out.description = node.description;
    if (node.enum) out.enum = node.enum;
    if (node.properties) {
        out.properties = Object.fromEntries(
            Object.entries(node.properties).map(([k, v]) => [k, toGeminiSchema(v)])
        );
    }
    if (node.items) out.items = toGeminiSchema(node.items);
    if (node.required?.length) out.required = node.required;
    return out;
}

/**
 * Build Gemini Live `functionDeclarations` plus a dispatcher for the tool calls
 * they produce.
 *
 * @param {string[]} [names] - tool names to expose; omit for all
 * @returns {{functionDeclarations: object[], dispatch: Function}}
 *
 * @example
 * const pdfTools = toGemini(['search_pdf', 'end_session']);
 * // config.tools: [{ functionDeclarations: pdfTools.functionDeclarations }, { googleSearch: {} }]
 * // on msg.toolCall: await pdfTools.dispatch(msg.toolCall.functionCalls, ctx)
 */
export function toGemini(names) {
    const entries = selectTools(names);

    const functionDeclarations = entries.map(([name, def]) => {
        const jsonSchema = z.toJSONSchema(def.schema);
        const parameters = toGeminiSchema(jsonSchema);
        // Gemini rejects an empty `properties` map; a bare OBJECT means "no args".
        if (!Object.keys(parameters.properties || {}).length) {
            delete parameters.properties;
            delete parameters.required;
        }
        return { name, description: def.description, parameters };
    });

    /**
     * Execute Gemini's functionCalls and shape them into the functionResponses
     * payload for `geminiSession.sendToolResponse()`.
     *
     * @param {Array<{id: string, name: string, args: object}>} functionCalls
     * @param {object} ctx - injected tool context
     * @returns {Promise<Array<{id, name, response: {output: string}}>>}
     */
    async function dispatch(functionCalls, ctx) {
        const known = new Set(entries.map(([name]) => name));
        return Promise.all(
            (functionCalls || []).map(async fn => {
                const output = known.has(fn.name)
                    ? await runTool(fn.name, fn.args, ctx)
                    : `Unknown tool: ${fn.name}`;
                return { id: fn.id, name: fn.name, response: { output } };
            })
        );
    }

    return { functionDeclarations, dispatch };
}
