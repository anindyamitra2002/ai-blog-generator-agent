"""Central configuration loaded from environment variables.

This module is the single source of truth for runtime settings — every other
module imports from here rather than reading ``os.environ`` directly. Load
your ``.env`` file by copying ``.env.example`` to ``.env`` and filling in the
keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from this file's directory).
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MEMORY_STORE_DIR = PROJECT_ROOT / "memory_store"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemini-2.5-flash:free")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Search APIs
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "2"))
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# ---------------------------------------------------------------------------
# Image APIs — priority order: Unsplash → Pexels → Pixabay
# ---------------------------------------------------------------------------
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")
IMAGE_MIN_WIDTH = int(os.getenv("IMAGE_MIN_WIDTH", "1200"))
IMAGE_MIN_HEIGHT = int(os.getenv("IMAGE_MIN_HEIGHT", "600"))

# ---------------------------------------------------------------------------
# Optional: Arxiv tool
# ---------------------------------------------------------------------------
ENABLE_ARXIV = os.getenv("ENABLE_ARXIV", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Memory settings (Mem0 / Chroma)
# ---------------------------------------------------------------------------
MEM0_USER_ID = os.getenv("MEM0_USER_ID", "blog-agent")


def validate() -> list[str]:
    """Return a list of human-readable configuration issues.

    Non-fatal — calling code should print these as warnings and continue.
    """
    issues: list[str] = []
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        issues.append(
            "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set. "
            "Please configure your OPENAI_API_KEY (e.g. from OpenRouter) in .env."
        )
    if not TAVILY_API_KEY and not SERPER_API_KEY:
        issues.append(
            "Neither TAVILY_API_KEY nor SERPER_API_KEY is set — deep search tools "
            "will be unavailable (Wikipedia and DuckDuckGo will still work)."
        )
    elif not TAVILY_API_KEY:
        issues.append(
            "TAVILY_API_KEY is not set — Tavily search tool will be unavailable."
        )
    elif not SERPER_API_KEY:
        issues.append(
            "SERPER_API_KEY is not set — Google Serper search tool will be unavailable."
        )
    if not (UNSPLASH_ACCESS_KEY or PEXELS_API_KEY or PIXABAY_API_KEY):
        issues.append(
            "No image API key configured (UNSPLASH_ACCESS_KEY / PEXELS_API_KEY / "
            "PIXABAY_API_KEY all empty) — cover image sourcing will fail. "
            "Use --skip-image to suppress this."
        )
    return issues


__all__ = [
    "PROJECT_ROOT",
    "OUTPUTS_DIR",
    "MEMORY_STORE_DIR",
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LLM_TEMPERATURE",
    "LLM_REQUEST_TIMEOUT",
    "TAVILY_API_KEY",
    "TAVILY_MAX_RESULTS",
    "SERPER_API_KEY",
    "UNSPLASH_ACCESS_KEY",
    "PEXELS_API_KEY",
    "PIXABAY_API_KEY",
    "IMAGE_MIN_WIDTH",
    "IMAGE_MIN_HEIGHT",
    "ENABLE_ARXIV",
    "MEM0_USER_ID",
    "validate",
]
