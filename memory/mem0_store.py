"""Mem0-backed memory layer for the blog agent.

Self-hosted Mem0 with a local Chroma vector store on disk — no external
service, no API key. Two operations are exposed:

  * ``check_similar(topic)`` — recall past posts on similar topics,
    so the agent can flag near-duplicates before regenerating.
  * ``record_post(...)``    — persist a completed post so future runs
    build a growing history the agent can draw on.

Scope is intentionally narrow for v1: this is NOT tracking evolving
facts or contradictions (that would be a Zep/temporal-graph use case),
just "have I covered this before, and where's the output?"
"""

from __future__ import annotations

from typing import Any, Optional

from config import (
    MEM0_USER_ID,
    MEMORY_STORE_DIR,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


class Mem0Store:
    """Thin wrapper around Mem0 for the blog agent's recall/record needs."""

    def __init__(self):
        # Import lazily so the rest of the project still imports cleanly
        # even if mem0ai isn't installed yet (helps during setup).
        try:
            from mem0 import Memory
        except ImportError as e:
            raise ImportError(
                "mem0ai is not installed. Run `pip install mem0ai` to enable memory."
            ) from e

        MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Build dynamic LLM configuration for Mem0
        if LLM_PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Please add your OpenRouter API key "
                    "to the OPENAI_API_KEY variable in your .env file."
                )
            llm_config = {
                "provider": "openai",
                "config": {
                    "model": OPENAI_MODEL,
                    "api_key": OPENAI_API_KEY,
                    "openai_base_url": OPENAI_API_BASE,
                }
            }
        else:
            llm_config = {
                "provider": "ollama",
                "config": {
                    "model": OLLAMA_MODEL,
                    "ollama_base_url": OLLAMA_BASE_URL,
                }
            }

        # Mem0 self-hosted config: Chroma vector store on local disk, with local Ollama/OpenAI LLM.
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "blog_agent_memory",
                    "path": str(MEMORY_STORE_DIR),
                },
            },
            "llm": llm_config,
            # Disable the LLM-based memory extraction step — we want to
            # store our own pre-formatted memory text verbatim rather
            # than letting Mem0 summarize it. This keeps the recall
            # signal predictable.
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5",
                },
            },
        }
        self._mem = Memory.from_config(config)
        self._user_id = MEM0_USER_ID

    def check_similar(
        self, topic: str, threshold: float = 0.4, limit: int = 5
    ) -> list[dict]:
        """Search memory for entries similar to the given topic.

        Returns a list of ``{"text": ..., "score": ...}`` dicts for
        entries above the similarity threshold. Higher threshold = stricter
        duplicate detection.
        """
        try:
            results: Any = self._mem.search(
                query=topic, filters={"user_id": self._user_id}, limit=limit
            )
        except Exception:
            try:
                results = self._mem.search(
                    query=topic, user_id=self._user_id, limit=limit
                )
            except Exception:
                return []

        # Mem0's return shape varies slightly across versions — normalize it.
        if isinstance(results, dict):
            items = results.get("results", [])
        elif isinstance(results, list):
            items = results
        else:
            items = []

        hits: list[dict] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            score = float(r.get("score", 0.0) or 0.0)
            if score < threshold:
                continue
            payload: Any = r.get("memory") or r.get("metadata", {}) or ""
            if isinstance(payload, dict):
                text = payload.get("text") or str(payload)
            else:
                text = str(payload)
            hits.append({"text": text, "score": score})
        return hits

    def record_post(
        self,
        topic: str,
        title: str,
        output_path: str,
        sources: list[str],
    ) -> Optional[str]:
        """Persist a completed post so future runs can recall it.

        Returns the memory ID assigned by Mem0, or None on failure.
        """
        memory_text = (
            f"Completed blog post on topic '{topic}'. "
            f"Title: '{title}'. "
            f"Output file: {output_path}. "
            f"Top sources: {', '.join(sources[:5]) if sources else '(none)'}"
        )
        try:
            result = self._mem.add(
                messages=[{"role": "user", "content": memory_text}],
                user_id=self._user_id,
                metadata={
                    "topic": topic,
                    "title": title,
                    "path": output_path,
                },
            )
        except Exception:
            return None

        # Best-effort extraction of the memory ID.
        if isinstance(result, dict):
            ids = result.get("results") or result.get("ids") or []
            if isinstance(ids, list) and ids:
                first = ids[0]
                if isinstance(first, dict):
                    return first.get("id")
                return str(first)
        return None


__all__ = ["Mem0Store"]
