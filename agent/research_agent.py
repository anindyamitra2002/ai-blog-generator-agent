"""Deep-research LangGraph agent.

Unlike a single-shot ``create_react_agent`` tool loop, this is an explicit
LangGraph ``StateGraph`` with four stages that repeat until the brief is
actually grounded in dated, recent information:

    plan -> search -> synthesize -> reflect --(gap found)--> plan (loop)
                                          \\--(no gap / out of budget)--> finalize -> END

* ``plan``       — LLM proposes 3-5 search queries for this iteration,
                    always anchored to today's real date, and always
                    including at least one explicitly "recent news" query.
* ``search``     — the planned queries are executed *deterministically*
                    (no further LLM tool-choice ambiguity) against the
                    matching search tool, in parallel.
* ``synthesize`` — LLM turns all collected search snippets into a single
                    structured research brief. It is instructed — via a
                    category-specific prompt file loaded from
                    ``prompts/research/<category>.md`` — to use ONLY the
                    raw evidence collected this run, never its own training
                    knowledge, and to say so explicitly when evidence is
                    thin rather than filling gaps with invented content.
* ``reflect``    — LLM checks its own brief: does it actually contain
                    dated, recent (last N days) facts with sources? If
                    not, it proposes follow-up queries and the graph loops
                    back to ``plan`` (bounded by DEEP_RESEARCH_MAX_ITERATIONS).
* ``finalize``   — stamps the brief with the generation date and extracts
                    the final source-URL list.

Category prompts (the "how to research this kind of topic" instructions,
including trusted domains and the anti-hallucination rules) live entirely
in ``prompts/research/*.md`` — one file per category — rather than being
hardcoded in this module. Edit those files to change how a category is
researched; this module only knows how to load and apply them.

This gives genuinely agentic, iterative "deep search" behavior instead of a
fixed 1-2-tool-call ceiling, while keeping each individual tool call
deterministic and cheap, and keeping every fact traceable to real search
evidence collected during the run.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

import config
from schemas import PlannedQuery
from tools.search_tools import build_search_toolkit

from .state import ResearchState

# ---------------------------------------------------------------------------
# Category list — kept as plain keys here (used only for classification
# matching and as filenames). The actual research instructions for each
# category live in prompts/research/<key>.md.
# ---------------------------------------------------------------------------
CATEGORY_KEYS: list[str] = [
    "geopolitics_international_relations",
    "national_news_politics",
    "artificial_intelligence_ml",
    "software_development_programming",
    "finance_investing_economics",
    "health_medicine_healthcare",
    "space_exploration_astronomy",
    "climate_change_environment",
    "legal_constitution_policy",
    "education_academia",
    "history_archaeology",
    "cryptocurrency_blockchain",
    "business_management_entrepreneurship",
    "cybersecurity_privacy",
    "automobile_transport",
    "gadgets_consumer_electronics",
    "agriculture_food_tech",
    "sports_athletics",
    "entertainment_media_pop_culture",
    "travel_tourism_hospitality",
    "sociology_anthropology_social_issues",
]

DEFAULT_CATEGORY = "geopolitics_international_relations"

PROMPTS_DIR = config.PROJECT_ROOT / "prompts" / "research"

# Below this many characters of *genuine* (non-error, non-empty) collected
# evidence, we don't trust the LLM to write a substantive brief without
# leaning on its own training data — so we flag the brief as thin/unverified
# and let main.py decide whether to warn the user or halt the pipeline.
MIN_EVIDENCE_CHARS = getattr(config, "MIN_EVIDENCE_CHARS", 400)


@lru_cache(maxsize=None)
def load_category_prompt(category: str) -> str:
    """Load the full research-instructions prompt for a category from disk.

    Falls back to the default category's file if the requested one is
    missing, so a typo'd/unknown category never crashes the pipeline.
    """
    path = PROMPTS_DIR / f"{category}.md"
    if not path.exists():
        path = PROMPTS_DIR / f"{DEFAULT_CATEGORY}.md"
    if not path.exists():
        # Last-resort inline fallback if the prompts directory itself is missing.
        return (
            "Focus on the topic directly. Use ONLY the raw search evidence "
            "provided below — never your own training knowledge. If evidence "
            "is thin, say so explicitly instead of inventing details."
        )
    return path.read_text(encoding="utf-8")


def _extract_directive_block(full_prompt: str) -> str:
    """Pull out just the short 'category + focus + trusted domains' block
    for use in the (shorter) planning prompt, so plan_node's context stays
    small even though synthesize_node uses the full file.
    """
    match = re.search(
        r"CRITICAL CATEGORY-SPECIFIC DIRECTIVE:(.*?)(?=\n=+\nCRITICAL GROUNDING)",
        full_prompt,
        re.DOTALL,
    )
    if match:
        return "CRITICAL CATEGORY-SPECIFIC DIRECTIVE:" + match.group(1).strip()
    return full_prompt[:600]


CATEGORIZATION_PROMPT_TEMPLATE = (
    "You are an expert content classifier. Given a blog topic, "
    "classify it into exactly one of the following 21 categories. Return only the "
    "exact category code name (e.g. 'artificial_intelligence_ml') and nothing else.\n\n"
    "Categories:\n"
    "1. geopolitics_international_relations\n"
    "2. national_news_politics\n"
    "3. artificial_intelligence_ml\n"
    "4. software_development_programming\n"
    "5. finance_investing_economics\n"
    "6. health_medicine_healthcare\n"
    "7. space_exploration_astronomy\n"
    "8. climate_change_environment\n"
    "9. legal_constitution_policy\n"
    "10. education_academia\n"
    "11. history_archaeology\n"
    "12. cryptocurrency_blockchain\n"
    "13. business_management_entrepreneurship\n"
    "14. cybersecurity_privacy\n"
    "15. automobile_transport\n"
    "16. gadgets_consumer_electronics\n"
    "17. agriculture_food_tech\n"
    "18. sports_athletics\n"
    "19. entertainment_media_pop_culture\n"
    "20. travel_tourism_hospitality\n"
    "21. sociology_anthropology_social_issues\n\n"
    "If the topic is a specific local/regional news event (a crime, accident, "
    "incident, or human-interest story tied to a place), classify it as "
    "'national_news_politics' unless it clearly fits a more specific category above.\n\n"
    "Topic: {topic}\n\n"
    "Response (category code name only):"
)


def categorize_topic(llm: BaseChatModel, topic: str) -> str:
    """Categorize the topic into one of the 21 categories."""
    prompt = CATEGORIZATION_PROMPT_TEMPLATE.format(topic=topic)
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip().lower().replace("'", "").replace('"', "")
        for cat in CATEGORY_KEYS:
            if cat in content:
                return cat
    except Exception as e:
        print(f"  - [warning] Categorization failed: {e}", flush=True)
    return DEFAULT_CATEGORY


# ---------------------------------------------------------------------------
# JSON helpers — LLM structured-output is prompted for, but parsed
# defensively since not every backend behind Omniroute's "auto" model
# guarantees strict JSON.
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end == -1 or end < start:
        return text
    return text[start : end + 1]


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(_extract_json_block(text))
    except Exception:
        return default


VALID_TOOLS = {
    "tavily_news", "tavily_deep", "serper_recent", "serper_deep",
    "exa_recent", "exa_deep", "linkup_recent", "linkup_deep",
    "wikipedia", "ddg_recent", "ddg_deep", "arxiv",
}

# DuckDuckGo is a fallback of last resort — never advertised to the planner
# LLM as a primary option when a stronger engine is configured.
_DDG_RECENT_PRIMARIES = ("tavily_news", "serper_recent", "exa_recent", "linkup_recent")
_DDG_DEEP_PRIMARIES = ("tavily_deep", "serper_deep", "exa_deep", "linkup_deep", "wikipedia")

# Regex to pull "[Title](URL)" markdown links out of the synthesized brief,
# so the Sources section can render real titles instead of bare URLs.
_MD_LINK_RE = re.compile(r"\[([^\[\]]{1,200}?)\]\((https?://[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s)\]\}<>]+")

# Substrings that mark a search result as a failure / non-answer rather than
# genuine evidence, so they don't count toward the "do we have enough real
# evidence" check and don't get treated as citable content.
_FAILURE_MARKERS = (
    "[tool error",
    "search failed",
    "search unavailable",
    "[error:",
)


def _is_genuine_evidence(content: str) -> bool:
    if not content or not content.strip():
        return False
    lowered = content.lower()
    return not any(marker in lowered for marker in _FAILURE_MARKERS)


def _coerce_queries(raw: Any, available: set[str], fallback_recent: str, fallback_deep: str) -> list[PlannedQuery]:
    """Turn loosely-parsed JSON into a validated list of PlannedQuery, remapping
    any tool name that isn't actually available (e.g. no TAVILY_API_KEY) to a
    working substitute so the plan never silently produces zero queries.

    Also demotes ddg_recent/ddg_deep back to a stronger primary tool
    whenever one is actually available — DuckDuckGo is fallback-of-last-
    resort only, even if the planner LLM picked it directly.
    """
    items = raw if isinstance(raw, list) else raw.get("queries", []) if isinstance(raw, dict) else []
    out: list[PlannedQuery] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        tool = str(item.get("tool", "")).strip()
        if tool not in VALID_TOOLS:
            tool = fallback_deep
        if tool not in available:
            is_recency = tool in ("tavily_news", "serper_recent", "exa_recent", "linkup_recent", "ddg_recent")
            tool = fallback_recent if is_recency else fallback_deep
        # DDG is a last resort: swap it out for a stronger primary tool if
        # one is configured, regardless of what the planner chose.
        if tool == "ddg_recent" and any(p in available for p in _DDG_RECENT_PRIMARIES):
            tool = fallback_recent
        elif tool == "ddg_deep" and any(p in available for p in _DDG_DEEP_PRIMARIES):
            tool = fallback_deep
        if tool not in available:
            continue
        out.append(PlannedQuery(tool=tool, query=query, reason=str(item.get("reason", ""))[:200]))
    return out[: config.MAX_QUERIES_PER_ITERATION]


def build_research_agent(llm: BaseChatModel, category: str = DEFAULT_CATEGORY) -> Any:
    """Construct and compile the deep-research LangGraph for the given category."""
    toolkit = build_search_toolkit(category)
    available = set(toolkit.keys())

    # Sensible fallbacks if the ideal tool isn't configured (no API key).
    # DuckDuckGo is deliberately last in both lists -- fallback of last resort.
    fallback_recent = next(
        (t for t in ("tavily_news", "serper_recent", "exa_recent", "linkup_recent", "ddg_recent") if t in available),
        "ddg_recent",
    )
    fallback_deep = next(
        (t for t in ("tavily_deep", "serper_deep", "exa_deep", "linkup_deep", "wikipedia", "ddg_deep") if t in available),
        "wikipedia",
    )

    full_category_prompt = load_category_prompt(category)
    category_directive = _extract_directive_block(full_category_prompt)

    # --- Node: plan ---------------------------------------------------------
    def plan_node(state: ResearchState) -> dict:
        iteration = state.get("iteration", 0)
        topic = state["topic"]
        today_human = state["today_human"]

        # DuckDuckGo is fallback-of-last-resort: hide it from the planner's
        # menu entirely unless it's the only tool available for that role,
        # so the LLM doesn't reach for it as a primary choice.
        planning_available = set(available)
        if any(p in available for p in _DDG_RECENT_PRIMARIES):
            planning_available.discard("ddg_recent")
        if any(p in available for p in _DDG_DEEP_PRIMARIES):
            planning_available.discard("ddg_deep")

        if iteration == 0:
            gap_note = ""
            n_queries = "6-8"
        else:
            missing = state.get("missing_aspects", [])
            gap_note = (
                "This is a FOLLOW-UP research round. The previous brief was still "
                "missing these specific things:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\nYour queries must target ONLY closing these gaps — do not "
                "re-research things already covered.\n\n"
            )
            n_queries = "3-4"

        prompt = f"""Today's date is {today_human}. You are planning search queries for a research brief on the topic: "{topic}".

{category_directive}

{gap_note}Available tools (use ONLY these exact names): {sorted(planning_available)}
  - tavily_news / serper_recent / exa_recent / linkup_recent = restricted to the last {config.NEWS_RECENCY_DAYS} days. Use for anything time-sensitive.
  - tavily_deep / serper_deep / exa_deep / linkup_deep / wikipedia = unrestricted background/definitional search.
  - arxiv = academic preprints (only if the topic is scientific/technical).
  - ddg_recent / ddg_deep, if listed, are a FALLBACK OF LAST RESORT — only use them if nothing else in the list above covers that role.

Generate a JSON array of {n_queries} search queries. Each item must be:
{{"tool": "<one of the available tool names above>", "query": "<search string>", "reason": "<short phrase>"}}

Rules:
- Spread queries across DIFFERENT tools (don't send every query to the same engine) — the goal is at least {config.MIN_SOURCES_TARGET} distinct, real source URLs collected across this iteration, and different engines surface different sources.
- At least one query MUST explicitly target the most recent news/developments — phrase it with words like "latest", "{today_human.split()[-1]}", or a specific recent month/year, and use a recency-restricted tool.
- Unless this is a follow-up round, include at least one background/definition query using an unrestricted tool.
- For a specific named person/place/incident, include the exact name/place as a standalone query too (not only broad category terms) — specific queries surface the real event, broad ones surface generic background instead.
- Do not duplicate query text across tools.
- Return ONLY the raw JSON array — no prose, no markdown code fences.
"""
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"  - [warning] Query planning failed: {e}", flush=True)
            content = "[]"

        parsed = _safe_json_loads(content, default=[])
        queries = _coerce_queries(parsed, available, fallback_recent, fallback_deep)

        if not queries:
            # Guaranteed minimal plan so the graph always makes forward progress.
            queries = [
                PlannedQuery(tool=fallback_recent, query=f"{topic} latest news {today_human}", reason="fallback recency query"),
                PlannedQuery(tool=fallback_deep, query=f"{topic}", reason="fallback background query"),
            ]

        for q in queries:
            print(f"  - [plan] ({q.tool}) {q.query}", flush=True)

        return {"planned_queries": queries}

    # --- Node: search --------------------------------------------------------
    def _run_one(pq: PlannedQuery) -> dict:
        tool_obj = toolkit.get(pq.tool)
        if tool_obj is None:
            return {"tool": pq.tool, "query": pq.query, "content": ""}
        try:
            result = tool_obj.invoke({"query": pq.query})
        except Exception as e:
            result = f"[tool error: {e}]"
        content = str(result)
        if len(content) > 4000:
            content = content[:4000] + " ...[truncated]"
        return {"tool": pq.tool, "query": pq.query, "content": content}

    def search_node(state: ResearchState) -> dict:
        planned = state.get("planned_queries", [])
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_SEARCHES) as ex:
            futures = {ex.submit(_run_one, pq): pq for pq in planned}
            for fut in as_completed(futures):
                pq = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"tool": pq.tool, "query": pq.query, "content": f"[error: {e}]"}
                print(f"  - [search] '{rec['tool']}' returned {len(rec['content'])} chars for: {rec['query']}", flush=True)
                results.append(rec)

        collected = list(state.get("collected", [])) + results
        return {"collected": collected}

    # --- Node: synthesize -----------------------------------------------------
    def synthesize_node(state: ResearchState) -> dict:
        topic = state["topic"]
        today_human = state["today_human"]
        collected = state.get("collected", [])

        genuine_records = [rec for rec in collected if _is_genuine_evidence(rec.get("content", ""))]
        genuine_chars = sum(len(rec["content"]) for rec in genuine_records)
        evidence_insufficient = genuine_chars < MIN_EVIDENCE_CHARS

        evidence_blocks = []
        for rec in collected:
            evidence_blocks.append(
                f"### Source call: {rec['tool']} | query: {rec['query']}\n{rec['content']}"
            )
        evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "(no search results collected)"

        thin_evidence_note = ""
        if evidence_insufficient:
            thin_evidence_note = (
                "\nNOTE: The raw evidence collected below is thin or largely empty "
                "(only "
                f"{genuine_chars} characters of genuine search content). This strongly "
                "suggests the search tools could not find substantive coverage of this "
                "exact topic. In this case you MUST NOT fill the gap with your own "
                "background knowledge. Write a short brief that honestly states what "
                "little (if anything) was found, and say explicitly that verified "
                "recent information could not be located — do not pad it out.\n"
            )

        distinct_urls = len({u for rec in genuine_records for u in _BARE_URL_RE.findall(rec["content"])})
        source_count_note = (
            f"\nCITATION BREADTH REQUIREMENT: The evidence collected this run contains "
            f"roughly {distinct_urls} distinct source URLs. Cite AT LEAST "
            f"{min(config.MIN_SOURCES_TARGET, distinct_urls)} of the distinct, genuinely "
            "different sources present in the evidence above — not just the first two or "
            "three. Every time you use a fact from a source, write it as a proper "
            "markdown link using that source's real title from its 'Title:' line where "
            "one is given (e.g. '[Reuters: India GDP grows 7%](https://reuters.com/...)'), "
            "falling back to the domain name as the title only if no title was given. Do "
            "NOT cite the same one or two sources for the whole brief while ignoring the "
            "rest of the evidence — spread citations across as many of the distinct "
            "sources collected as actually contain relevant, usable information.\n"
        )

        # Full per-category instructions (focus, trusted domains, structure,
        # and — critically — the anti-hallucination / grounding rules) come
        # straight from the prompts/research/<category>.md file.
        prompt = f"""Today's date is {today_human}. You are compiling a research brief on: "{topic}".

{full_category_prompt}
{thin_evidence_note}
{source_count_note}
Raw evidence collected this run (this is the ONLY information you may use):
{evidence_text}

Write the brief now, following the OUTPUT FORMAT given above exactly, with
{{today_human}} replaced by {today_human}. Only include the brief itself —
no preamble, no meta-commentary.
"""
        try:
            response = llm.invoke(prompt)
            brief = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"  - [warning] Synthesis failed: {e}", flush=True)
            brief = evidence_text

        if evidence_insufficient:
            print(
                f"  - [synthesize] [warning] Evidence is thin ({genuine_chars} genuine "
                "chars) — brief flagged as evidence_insufficient.",
                flush=True,
            )

        return {"brief": brief, "evidence_insufficient": evidence_insufficient}

    # --- Node: reflect ---------------------------------------------------------
    def reflect_node(state: ResearchState) -> dict:
        iteration = state.get("iteration", 0) + 1
        today_human = state["today_human"]
        brief = state.get("brief", "")
        collected = state.get("collected", [])

        # How many distinct sources does the brief actually cite, vs. how
        # many distinct sources are even available in the raw evidence
        # collected so far? We only ask for more if there's real headroom.
        cited_urls = {url for _, url in _MD_LINK_RE.findall(brief)} | set(_BARE_URL_RE.findall(brief))
        available_urls = {
            u
            for rec in collected
            if _is_genuine_evidence(rec.get("content", ""))
            for u in _BARE_URL_RE.findall(rec["content"])
        }
        source_target = min(config.MIN_SOURCES_TARGET, len(available_urls)) if available_urls else 0
        source_count_ok = len(cited_urls) >= source_target

        prompt = f"""Today's date is {today_human}. Review this research brief:

---
{brief}
---

Does the "Recent developments" section contain SPECIFIC, DATED facts or
events from roughly the last {config.NEWS_RECENCY_DAYS}-{config.NEWS_RECENCY_DAYS * 2} days, each with a source URL that appears verbatim in the brief?
(Generic, undated statements like "the situation continues to evolve", or an
explicit "no verified recent developments were found" statement, do NOT count
as satisfied — but if the brief HONESTLY says nothing recent was found, do not
ask for follow-up queries that were already tried; instead accept the honest
gap unless there are genuinely untried angles.)

Return ONLY a JSON object, no prose:
{{"has_recent_dated_facts": true or false, "missing_aspects": ["short phrase", ...], "follow_up_queries": []}}

missing_aspects should be empty if has_recent_dated_facts is true.
"""
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"  - [warning] Reflection failed: {e}", flush=True)
            content = '{"has_recent_dated_facts": true, "missing_aspects": []}'

        parsed = _safe_json_loads(content, default={"has_recent_dated_facts": True, "missing_aspects": []})
        has_recent = bool(parsed.get("has_recent_dated_facts", True))
        missing = parsed.get("missing_aspects", []) or []
        if not isinstance(missing, list):
            missing = [str(missing)]

        if not source_count_ok:
            missing = missing + [
                f"Not enough distinct sources cited ({len(cited_urls)}/{source_target} target) — "
                "cite more of the distinct sources already present in the collected evidence, "
                "and/or search for additional angles with different tools to surface new sources."
            ]

        print(
            f"  - [reflect] iteration={iteration} has_recent_dated_facts={has_recent} "
            f"cited_sources={len(cited_urls)}/{source_target} missing={missing}",
            flush=True,
        )

        return {
            "iteration": iteration,
            "has_recent_dated_facts": has_recent,
            "missing_aspects": missing,
            "source_count_ok": source_count_ok,
        }

    def route_after_reflect(state: ResearchState) -> str:
        recent_ok = state.get("has_recent_dated_facts", True)
        sources_ok = state.get("source_count_ok", True)
        if recent_ok and sources_ok:
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", config.DEEP_RESEARCH_MAX_ITERATIONS):
            print("  - [reflect] max iterations reached, finalizing with best available brief.", flush=True)
            return "end"
        return "continue"

    # --- Node: finalize ---------------------------------------------------------
    def _title_from_raw_evidence(url: str, collected: list) -> str:
        """Best-effort title lookup for a bare URL from the 'Title: ...'
        lines the exa/linkup tools emit next to each result's URL."""
        for rec in collected:
            content = rec.get("content", "")
            if url not in content:
                continue
            for block in content.split("\n\n"):
                if url in block:
                    m = re.search(r"^Title:\s*(.+)$", block, re.MULTILINE)
                    if m:
                        return m.group(1).strip()
        return ""

    def finalize_node(state: ResearchState) -> dict:
        brief = state.get("brief", "")
        today_human = state["today_human"]
        collected = state.get("collected", [])

        # --- 1. Titled sources actually cited as markdown links in the brief.
        sources: list[dict] = []
        seen: set[str] = set()
        for title, url in _MD_LINK_RE.findall(brief):
            u = url.rstrip(".,;:!?")
            if u not in seen:
                seen.add(u)
                sources.append({"title": title.strip(), "url": u})

        # --- 2. Any remaining bare URLs mentioned in the brief but not
        # already wrapped in a markdown link — give them a best-effort
        # title from the raw evidence, falling back to the domain name.
        for u in _BARE_URL_RE.findall(brief):
            u = u.rstrip(".,;:!?")
            if u in seen:
                continue
            seen.add(u)
            title = _title_from_raw_evidence(u, collected)
            if not title:
                from urllib.parse import urlparse

                title = urlparse(u).netloc or u
            sources.append({"title": title, "url": u})

        # --- 3. Pad up to MIN_SOURCES_TARGET from raw collected evidence
        # (genuine, non-error results only) whenever the brief itself
        # under-cites relative to what was actually collected this run.
        # Two passes: first the structured "Title: ... / URL: ..." blocks
        # that exa/linkup emit (best titles), then any remaining bare URLs
        # from tools with unstructured output (tavily/serper/wikipedia/ddg),
        # using the domain name as the title.
        if len(sources) < config.MIN_SOURCES_TARGET:
            for rec in collected:
                content = rec.get("content", "")
                if not _is_genuine_evidence(content):
                    continue
                for block in content.split("\n\n"):
                    if len(sources) >= config.MIN_SOURCES_TARGET:
                        break
                    url_match = re.search(r"URL:\s*(https?://\S+)", block)
                    if not url_match:
                        continue
                    u = url_match.group(1).rstrip(".,;:!?")
                    if u in seen:
                        continue
                    seen.add(u)
                    title_match = re.search(r"^Title:\s*(.+)$", block, re.MULTILINE)
                    title = title_match.group(1).strip() if title_match else u
                    sources.append({"title": title, "url": u})
                if len(sources) >= config.MIN_SOURCES_TARGET:
                    break

            if len(sources) < config.MIN_SOURCES_TARGET:
                from urllib.parse import urlparse

                for rec in collected:
                    content = rec.get("content", "")
                    if not _is_genuine_evidence(content):
                        continue
                    for u in _BARE_URL_RE.findall(content):
                        if len(sources) >= config.MIN_SOURCES_TARGET:
                            break
                        u = u.rstrip(".,;:!?")
                        if u in seen:
                            continue
                        seen.add(u)
                        sources.append({"title": urlparse(u).netloc or u, "url": u})
                    if len(sources) >= config.MIN_SOURCES_TARGET:
                        break

        source_urls = [s["url"] for s in sources]

        header = f"_Research compiled on {today_human}. Recency window: last {config.NEWS_RECENCY_DAYS} days for news._\n\n"
        final_brief = header + brief.strip()

        print(f"  - [finalize] {len(sources)} distinct source(s) collected for citation.", flush=True)

        return {"brief": final_brief, "source_urls": source_urls, "sources": sources}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {"continue": "plan", "end": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()


def run_research(agent: Any, topic: str) -> dict:
    """Run the deep-research graph and return the full final state dict.

    Callers should read at least ``brief`` and ``evidence_insufficient`` —
    the latter is True when the search tools couldn't find substantive
    coverage of the topic, which callers should treat as a strong signal to
    warn the user (or abort) rather than silently generating a post that
    may lean on the LLM's own (possibly stale or wrong) background knowledge.
    """
    print("  - Starting deep research graph...", flush=True)

    initial_state: ResearchState = {
        "topic": topic,
        "today": config.TODAY_STR,
        "today_human": config.TODAY_HUMAN,
        "iteration": 0,
        "max_iterations": config.DEEP_RESEARCH_MAX_ITERATIONS,
        "planned_queries": [],
        "collected": [],
        "brief": "",
        "has_recent_dated_facts": False,
        "missing_aspects": [],
        "source_count_ok": False,
        "source_urls": [],
        "sources": [],
        "evidence_insufficient": False,
    }

    final_state: dict = dict(initial_state)
    try:
        for step in agent.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, node_state in step.items():
                print(f"  - [graph] completed node: {node_name}", flush=True)
                final_state.update(node_state or {})
    except Exception as e:
        print(f"  - [warning] Research graph error: {e}", flush=True)

    return final_state


def get_source_urls(agent_result_brief: str) -> list[str]:
    """Convenience helper if a caller only has the brief text on hand."""
    urls = re.findall(r"https?://[^\s)\]\}<>]+", agent_result_brief)
    cleaned: list[str] = []
    seen: set[str] = set()
    for u in urls:
        u = u.rstrip(".,;:!?")
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


__all__ = [
    "build_research_agent",
    "run_research",
    "categorize_topic",
    "get_source_urls",
    "load_category_prompt",
    "CATEGORY_KEYS",
    "DEFAULT_CATEGORY",
]