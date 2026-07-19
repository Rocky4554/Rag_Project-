/**
 * Shared tool registry.
 *
 * Import adapters directly rather than from here — each pulls in its own heavy
 * framework dependency, and the Gemini voice worker should not have to load
 * LangChain to run:
 *
 *   import { toGemini }    from '../../lib/tools/adapters/gemini.js';
 *   import { toLiveKit }   from '../../lib/tools/adapters/livekit.js';
 *   import { toLangChain } from '../../lib/tools/adapters/langchain.js';
 */
export { tools, selectTools, runTool } from './registry.js';
