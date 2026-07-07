"""Shared state definition for the LangGraph deep-research graph."""

from __future__ import annotations

from typing import TypedDict

from schemas import PlannedQuery


class SearchRecord(TypedDict):
    tool: str
    query: str
    content: str


class ResearchState(TypedDict, total=False):
    topic: str
    category: str
    today: str
    today_human: str

    iteration: int
    max_iterations: int

    planned_queries: list[PlannedQuery]
    collected: list[SearchRecord]

    brief: str
    has_recent_dated_facts: bool
    missing_aspects: list[str]
    # True once the brief cites at least min(MIN_SOURCES_TARGET, distinct
    # sources actually available in collected evidence) distinct sources.
    source_count_ok: bool

    source_urls: list[str]

    # Structured title+url pairs (see schemas.Source), deduplicated and
    # padded (from raw collected evidence) up to config.MIN_SOURCES_TARGET
    # whenever the LLM-written brief itself cites fewer. This is what the
    # assembler renders as proper markdown links in the Sources section,
    # instead of bare/raw URLs.
    sources: list[dict]

    # True when the search tools returned too little genuine content to
    # ground a real brief in. Downstream stages / main.py use this to warn
    # the user (or halt) instead of silently generating a post that may be
    # leaning on the LLM's own background knowledge rather than real search
    # results.
    evidence_insufficient: bool


__all__ = ["ResearchState", "SearchRecord"]