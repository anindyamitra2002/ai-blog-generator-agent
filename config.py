"""Central configuration loaded from environment variables.

This module is the single source of truth for runtime settings — every other
module imports from here rather than reading ``os.environ`` directly. Load
your ``.env`` file by copying ``.env.example`` to ``.env`` and filling in the
keys.
"""

import datetime as _dt
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
# Date / recency context — computed fresh every run, never hardcoded.
# Every prompt in the pipeline (research, outline, writer, editor) is given
# this so the model never silently falls back on its training cutoff when
# talking about "today", "this year", or "recent" events.
# ---------------------------------------------------------------------------
TODAY: _dt.date = _dt.date.today()
TODAY_STR: str = TODAY.strftime("%Y-%m-%d")
TODAY_HUMAN: str = TODAY.strftime("%B %d, %Y")  # e.g. "July 07, 2026"
CURRENT_YEAR: int = TODAY.year
CURRENT_MONTH_YEAR: str = TODAY.strftime("%B %Y")  # e.g. "July 2026"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# LLM_PROVIDER "openai" is used both for real OpenAI and for any
# OpenAI-compatible proxy — including Omniroute, which is what this project
# is wired to by default (OPENAI_API_BASE points at the local Omniroute
# router and OPENAI_MODEL="auto" lets Omniroute pick the best backing model).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemini-2.5-flash:free")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# Low-temperature LLM settings for structured/deterministic sub-tasks inside
# the research graph (query planning, reflection/gap-checking). Using a
# separate lower temperature keeps JSON planning steps reliable even when
# the main writing temperature is creative.
PLANNER_TEMPERATURE = float(os.getenv("PLANNER_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Search APIs
# ---------------------------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# ---------------------------------------------------------------------------
# Deep-research / recency settings
# ---------------------------------------------------------------------------
# How far back a query counts as "recent news" (used for Tavily's news topic
# `days` parameter, DDG's `time` window, and Serper's `tbs` window).
NEWS_RECENCY_DAYS = int(os.getenv("NEWS_RECENCY_DAYS", "14"))
# How far back the *background* research is allowed to range unrestricted.
BACKGROUND_RECENCY_DAYS = int(os.getenv("BACKGROUND_RECENCY_DAYS", "365"))
# Max number of plan -> search -> reflect loops in the deep research graph.
DEEP_RESEARCH_MAX_ITERATIONS = int(os.getenv("DEEP_RESEARCH_MAX_ITERATIONS", "3"))
# Max number of queries planned per iteration (across all tools).
MAX_QUERIES_PER_ITERATION = int(os.getenv("MAX_QUERIES_PER_ITERATION", "5"))
# Max parallel tool calls executed concurrently within a single iteration.
MAX_PARALLEL_SEARCHES = int(os.getenv("MAX_PARALLEL_SEARCHES", "5"))

# ---------------------------------------------------------------------------
# Image APIs — priority order: Serper/Tavily (web) → Unsplash → Pexels → Pixabay
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
            "Please configure your OPENAI_API_KEY (e.g. your Omniroute key) in .env."
        )
    if not TAVILY_API_KEY and not SERPER_API_KEY:
        issues.append(
            "Neither TAVILY_API_KEY nor SERPER_API_KEY is set — deep search and "
            "recent-news tools will be unavailable (Wikipedia and DuckDuckGo will "
            "still work, but recency will suffer)."
        )
    elif not TAVILY_API_KEY:
        issues.append(
            "TAVILY_API_KEY is not set — Tavily search/news tools will be unavailable."
        )
    elif not SERPER_API_KEY:
        issues.append(
            "SERPER_API_KEY is not set — Google Serper search tool will be unavailable."
        )
    if not (UNSPLASH_ACCESS_KEY or PEXELS_API_KEY or PIXABAY_API_KEY):
        issues.append(
            "No stock-photo API key configured (UNSPLASH_ACCESS_KEY / PEXELS_API_KEY / "
            "PIXABAY_API_KEY all empty) — cover image sourcing will fall back to "
            "Serper/Tavily web image search only. Use --skip-image to suppress this."
        )
    return issues


__all__ = [
    "PROJECT_ROOT",
    "OUTPUTS_DIR",
    "MEMORY_STORE_DIR",
    "TODAY",
    "TODAY_STR",
    "TODAY_HUMAN",
    "CURRENT_YEAR",
    "CURRENT_MONTH_YEAR",
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LLM_TEMPERATURE",
    "PLANNER_TEMPERATURE",
    "LLM_REQUEST_TIMEOUT",
    "TAVILY_API_KEY",
    "TAVILY_MAX_RESULTS",
    "SERPER_API_KEY",
    "NEWS_RECENCY_DAYS",
    "BACKGROUND_RECENCY_DAYS",
    "DEEP_RESEARCH_MAX_ITERATIONS",
    "MAX_QUERIES_PER_ITERATION",
    "MAX_PARALLEL_SEARCHES",
    "UNSPLASH_ACCESS_KEY",
    "PEXELS_API_KEY",
    "PIXABAY_API_KEY",
    "IMAGE_MIN_WIDTH",
    "IMAGE_MIN_HEIGHT",
    "ENABLE_ARXIV",
    "MEM0_USER_ID",
    "validate",
]
