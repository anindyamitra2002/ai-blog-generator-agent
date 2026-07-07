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

    source_urls: list[str]


__all__ = ["ResearchState", "SearchRecord"]
