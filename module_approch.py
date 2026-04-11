"""
livekit_agent_session.py — Modern LiveKit Agents v1.x Voice Pipeline (Python)

Uses the current AgentSession API (VoicePipelineAgent is deprecated in 1.x).

Run:
    python livekit_agent_session.py dev

Required env vars:
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    DEEPGRAM_API_KEY          (for STT)
    OPENAI_API_KEY            (for LLM)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# LiveKit Agents v1.x SDK
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)

# Plugins
from livekit.plugins import deepgram, openai, silero

# LiveKit Turn Detector — uses an ML model to determine when the user
# has finished their thought (not just stopped making sound)
from livekit.plugins.turn_detector import MultilingualModel

# Noise Cancellation — filters background noise before audio hits VAD/STT
from livekit.agents.noise_cancellation import BVC

# Load environment variables
load_dotenv()

# Logger
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — Select providers via env vars
# ═══════════════════════════════════════════════════════════════════

STT_MODEL = os.getenv("LK_STT_MODEL", "nova-3")
STT_LANGUAGE = os.getenv("LK_STT_LANGUAGE", "en")
LLM_MODEL = os.getenv("LK_LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LK_LLM_PROVIDER", "openai")  # 'openai' | 'google'
TTS_MODEL = os.getenv("LK_TTS_MODEL", "aura-2-asteria-en")
TTS_PROVIDER = os.getenv("LK_TTS_PROVIDER", "deepgram")  # 'deepgram' | 'openai'


# ═══════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS — Create STT, LLM, TTS instances
# ═══════════════════════════════════════════════════════════════════

def create_stt():
    """Create the STT (Speech-to-Text) instance."""
    return deepgram.STT(
        model=STT_MODEL,
        language=STT_LANGUAGE,
    )


def create_llm(model_override: Optional[str] = None):
    """Create the LLM instance."""
    model = model_override or LLM_MODEL

    if LLM_PROVIDER == "google":
        try:
            from livekit.plugins import google
            return google.LLM(model=model)
        except ImportError:
            raise ImportError(
                "Google LLM plugin requires livekit-plugins-google. "
                "Install with: pip install livekit-plugins-google"
            )

    # Default: OpenAI
    return openai.LLM(model=model)


def create_tts():
    """Create the TTS (Text-to-Speech) instance."""
    if TTS_PROVIDER == "openai":
        return openai.TTS(
            model=os.getenv("LK_TTS_OPENAI_MODEL", "tts-1"),
            voice=os.getenv("LK_TTS_OPENAI_VOICE", "alloy"),
        )

    # Default: Deepgram
    return deepgram.TTS(model=TTS_MODEL)


# ═══════════════════════════════════════════════════════════════════
# AGENT CLASS — Your voice AI personality and tools
# ═══════════════════════════════════════════════════════════════════

class VoiceAgent(Agent):
    """
    A reusable voice agent. Extend this class to add tools.

    Example with tools:
        class MyAgent(VoiceAgent):
            def __init__(self):
                super().__init__(
                    instructions="You help users search documents.",
                    tools=[search_pdf_tool],
                )
    """

    def __init__(
        self,
        instructions: str = "You are a helpful, friendly voice AI assistant. Respond concisely and conversationally.",
        tools: Optional[list] = None,
    ):
        super().__init__(
            instructions=instructions,
            tools=tools or [],
        )


# ═══════════════════════════════════════════════════════════════════
# PREWARM — Load heavy models once when the worker starts
# ═══════════════════════════════════════════════════════════════════

def prewarm(proc: JobProcess):
    """Load Silero VAD model once, shared across all jobs."""
    logger.info("Prewarming: Loading Silero VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete")


# ═══════════════════════════════════════════════════════════════════
# ENTRYPOINT — Called for each new room/job
# ═══════════════════════════════════════════════════════════════════

async def entrypoint(ctx: JobContext):
    """Main entry point for each voice agent job."""
    logger.info(f"Agent job received for room: {ctx.room.name}")

    # 1. Create the Agent
    agent = VoiceAgent()

    # 2. Create the AgentSession with the full pipeline
    session = AgentSession(
        stt=create_stt(),
        llm=create_llm(),
        tts=create_tts(),
        vad=ctx.proc.userdata["vad"],
        # LiveKit Multilingual Turn Detector — ML model that understands
        # when the user has finished a complete statement, even with pauses.
        # This is the game-changer for natural conversation flow.
        turn_detection=MultilingualModel(),
    )

    # 3. Start the session with noise cancellation
    await session.start(
        room=ctx.room,
        agent=agent,
        # Noise cancellation runs BEFORE audio hits Silero VAD and Deepgram STT.
        # Huge help for accents, quiet speech, and noisy environments.
        input_options={
            "noise_cancellation": BVC(),
        },
    )

    # 4. Connect to the room
    await ctx.connect()
    logger.info(f"Agent connected to room: {ctx.room.name}")

    # 5. Greet the user
    await session.generate_reply(
        instructions="Greet the user warmly and let them know you are ready to help."
    )


# ═══════════════════════════════════════════════════════════════════
# CLI RUNNER — python livekit_agent_session.py dev
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
