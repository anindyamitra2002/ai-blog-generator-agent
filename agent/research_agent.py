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
                    structured research brief, instructed to keep every
                    date/statistic/URL it finds.
* ``reflect``    — LLM checks its own brief: does it actually contain
                    dated, recent (last N days) facts with sources? If
                    not, it proposes follow-up queries and the graph loops
                    back to ``plan`` (bounded by DEEP_RESEARCH_MAX_ITERATIONS).
* ``finalize``   — stamps the brief with the generation date and extracts
                    the final source-URL list.

This gives genuinely agentic, iterative "deep search" behavior instead of a
fixed 1-2-tool-call ceiling, while keeping each individual tool call
deterministic and cheap.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

import config
from schemas import PlannedQuery
from tools.search_tools import build_search_toolkit

from .state import ResearchState

DEFAULT_CATEGORY = "geopolitics_international_relations"

CATEGORY_INSTRUCTIONS = {
    "geopolitics_international_relations": (
        "Focus on international relations, foreign policy, diplomacy, and global treaties. "
        "Your sources must be primarily major Indian news portals (ABP, NDTV, ANI, PTI), official government publications (PIB, Ministry of External Affairs), or established policy think-tanks."
    ),
    "national_news_politics": (
        "Focus on domestic politics, governance, and national events. "
        "Your sources must be major Indian news portals (ABP, NDTV, ANI, PTI) and government press releases."
    ),
    "artificial_intelligence_ml": (
        "Focus on academic advancements in AI, machine learning models, and datasets. "
        "Your sources must be academic papers (arXiv), official research blogs, and developer hubs (HuggingFace, PapersWithCode)."
    ),
    "software_development_programming": (
        "Focus on coding tutorials, API references, software practices, and package documentation. "
        "Your sources must be official docs, MDN, StackOverflow, GitHub repositories, or developer guides."
    ),
    "finance_investing_economics": (
        "Focus on business, economic policy, market movements, and financial reports. "
        "Your sources must be major financial publications (Economic Times, Livemint, Business Standard) and regulatory portals (RBI, Ministry of Finance)."
    ),
    "health_medicine_healthcare": (
        "Focus on medical research, public health advisories, and bioscience. "
        "Your sources must be trusted organizations (WHO), peer-reviewed research (PubMed, Lancet), and public health portals (Ministry of Health, ICMR)."
    ),
    "space_exploration_astronomy": (
        "Focus on space missions, astronomical events, and space technology. "
        "Your sources must be official space agencies (ISRO, NASA, ESA), astronomy magazines, or astrophysics preprints."
    ),
    "climate_change_environment": (
        "Focus on ecology, climate change science, and environment policy. "
        "Your sources must be trusted bodies (IPCC, UNEP), scientific journals, and environment ministries."
    ),
    "legal_constitution_policy": (
        "Focus on public policy, judicial rulings, legal updates, and constitutional law. "
        "Your sources must be official court portals, trusted legal digests, legislative updates, and think-tanks."
    ),
    "education_academia": (
        "Focus on academic curriculum, educational technology, learning guides, and career counseling. "
        "Your sources must be official academic regulators (Ministry of Education, UGC, AICTE), universities, or scholar search engines."
    ),
    "history_archaeology": (
        "Focus on historical analysis, heritage sites, and archeological findings. "
        "Your sources must be archaeological surveys, academic databases, history catalogs, and museums."
    ),
    "cryptocurrency_blockchain": (
        "Focus on Web3 projects, blockchain protocols, and crypto market insights. "
        "Your sources must be developer sites, cryptographic research, or leading crypto outlets."
    ),
    "business_management_entrepreneurship": (
        "Focus on business strategy, entrepreneurship, and management principles. "
        "Your sources must be major startup portals, business reviews (HBR), and entrepreneur guides."
    ),
    "cybersecurity_privacy": (
        "Focus on cyber threats, security practices, privacy laws, and software vulnerabilities. "
        "Your sources must be official response teams (CERT-In), security groups (SANS, OWASP), and trusted cybersecurity journals."
    ),
    "automobile_transport": (
        "Focus on electric vehicles, automotive technology, and transit systems. "
        "Your sources must be autotech journals, vehicle portals, and transport directories."
    ),
    "gadgets_consumer_electronics": (
        "Focus on mobile reviews, home electronics, and consumer hardware news. "
        "Your sources must be verified gadget databases, consumer tech magazines, and electronics guides."
    ),
    "agriculture_food_tech": (
        "Focus on smart farming, crops, agrochemicals, and food science. "
        "Your sources must be agricultural bodies (ICAR), food directories, and agriculture magazines."
    ),
    "sports_athletics": (
        "Focus on cricket, athletics, and international sports tournaments. "
        "Your sources must be sports news networks (Cricinfo, Sportstar, NDTV Sports) and official athletic boards."
    ),
    "entertainment_media_pop_culture": (
        "Focus on movies, music, popular culture, and streaming platforms. "
        "Your sources must be film reviews, major entertainment media outlets, and pop-culture portals."
    ),
    "travel_tourism_hospitality": (
        "Focus on travel itineraries, destinations, cultural tourism, and hospitality trends. "
        "Your sources must be tourism boards (Incredible India), verified travel guides, and travelers forums."
    ),
    "sociology_anthropology_social_issues": (
        "Focus on public policy, social studies, demographics, and human rights. "
        "Your sources must be international reports (UN, World Bank), think-tanks, and sociology journals."
    ),
}

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
        for cat in CATEGORY_INSTRUCTIONS.keys():
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
    "wikipedia", "ddg_recent", "ddg_deep", "arxiv",
}


def _coerce_queries(raw: Any, available: set[str], fallback_recent: str, fallback_deep: str) -> list[PlannedQuery]:
    """Turn loosely-parsed JSON into a validated list of PlannedQuery, remapping
    any tool name that isn't actually available (e.g. no TAVILY_API_KEY) to a
    working substitute so the plan never silently produces zero queries.
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
            is_recency = tool in ("tavily_news", "serper_recent", "ddg_recent")
            tool = fallback_recent if is_recency else fallback_deep
        if tool not in available:
            continue
        out.append(PlannedQuery(tool=tool, query=query, reason=str(item.get("reason", ""))[:200]))
    return out[: config.MAX_QUERIES_PER_ITERATION]


def build_research_agent(llm: BaseChatModel, category: str = DEFAULT_CATEGORY) -> Any:
    """Construct and compile the deep-research LangGraph for the given category."""
    toolkit = build_search_toolkit(category)
    available = set(toolkit.keys())

    # Sensible fallbacks if the ideal tool isn't configured (no API key).
    fallback_recent = next((t for t in ("tavily_news", "serper_recent", "ddg_recent") if t in available), "ddg_recent")
    fallback_deep = next((t for t in ("tavily_deep", "serper_deep", "wikipedia", "ddg_deep") if t in available), "wikipedia")

    category_instructions = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS[DEFAULT_CATEGORY])

    # --- Node: plan ---------------------------------------------------------
    def plan_node(state: ResearchState) -> dict:
        iteration = state.get("iteration", 0)
        topic = state["topic"]
        today_human = state["today_human"]

        if iteration == 0:
            gap_note = ""
            n_queries = "3-5"
        else:
            missing = state.get("missing_aspects", [])
            gap_note = (
                "This is a FOLLOW-UP research round. The previous brief was still "
                "missing these specific things:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\nYour queries must target ONLY closing these gaps — do not "
                "re-research things already covered.\n\n"
            )
            n_queries = "2-3"

        prompt = f"""Today's date is {today_human}. You are planning search queries for a research brief on the topic: "{topic}".

{category_instructions}

{gap_note}Available tools (use ONLY these exact names): {sorted(available)}
  - tavily_news / serper_recent / ddg_recent = restricted to the last {config.NEWS_RECENCY_DAYS} days. Use for anything time-sensitive.
  - tavily_deep / serper_deep / ddg_deep / wikipedia = unrestricted background/definitional search.
  - arxiv = academic preprints (only if the topic is scientific/technical).

Generate a JSON array of {n_queries} search queries. Each item must be:
{{"tool": "<one of the available tool names above>", "query": "<search string>", "reason": "<short phrase>"}}

Rules:
- At least one query MUST explicitly target the most recent news/developments — phrase it with words like "latest", "{today_human.split()[-1]}", or a specific recent month/year, and use a recency-restricted tool.
- Unless this is a follow-up round, include at least one background/definition query using an unrestricted tool.
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
                PlannedQuery(tool=fallback_recent, query=f"latest {topic} news {today_human}", reason="fallback recency query"),
                PlannedQuery(tool=fallback_deep, query=f"{topic} overview", reason="fallback background query"),
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

        evidence_blocks = []
        for rec in collected:
            evidence_blocks.append(
                f"### Source call: {rec['tool']} | query: {rec['query']}\n{rec['content']}"
            )
        evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "(no search results collected)"

        prompt = f"""Today's date is {today_human}. You are compiling a research brief on: "{topic}".

{category_instructions}

Below are raw search results gathered from multiple tools/queries. Synthesize
them into ONE consolidated, well-organized research brief. CRITICAL:
- Explicitly call out dates whenever a source result mentions one (e.g. "on
  3 July 2026, ..."). Never blur a dated fact into a vague "recently".
- Preserve every concrete statistic, name, and specific example you find.
- Keep the source URL next to each fact you cite from it.
- If the evidence contains conflicting recent claims, note the discrepancy
  rather than silently picking one.
- Do not invent facts not present in the evidence below.

Raw evidence:
{evidence_text}

Write the brief in exactly this structure:

## Definition
(short paragraph)

## Key subtopics
- Subtopic A — one-line description
- Subtopic B — one-line description

## Recent developments / examples (as of {today_human})
- Dated fact or example, with source URL in parentheses
- ...

## Notable sources
- URL 1
- URL 2

Only include the brief itself — no preamble, no meta-commentary.
"""
        try:
            response = llm.invoke(prompt)
            brief = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"  - [warning] Synthesis failed: {e}", flush=True)
            brief = evidence_text

        return {"brief": brief}

    # --- Node: reflect ---------------------------------------------------------
    def reflect_node(state: ResearchState) -> dict:
        iteration = state.get("iteration", 0) + 1
        today_human = state["today_human"]
        brief = state.get("brief", "")

        prompt = f"""Today's date is {today_human}. Review this research brief:

---
{brief}
---

Does the "Recent developments" section contain SPECIFIC, DATED facts or
events from roughly the last {config.NEWS_RECENCY_DAYS}-{config.NEWS_RECENCY_DAYS * 2} days, each with a source URL?
(Generic, undated statements like "the situation continues to evolve" do NOT count.)

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

        print(f"  - [reflect] iteration={iteration} has_recent_dated_facts={has_recent} missing={missing}", flush=True)

        return {"iteration": iteration, "has_recent_dated_facts": has_recent, "missing_aspects": missing}

    def route_after_reflect(state: ResearchState) -> str:
        if state.get("has_recent_dated_facts", True):
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", config.DEEP_RESEARCH_MAX_ITERATIONS):
            print("  - [reflect] max iterations reached, finalizing with best available brief.", flush=True)
            return "end"
        return "continue"

    # --- Node: finalize ---------------------------------------------------------
    def finalize_node(state: ResearchState) -> dict:
        brief = state.get("brief", "")
        today_human = state["today_human"]

        urls = re.findall(r"https?://[^\s)\]\}<>]+", brief)
        cleaned: list[str] = []
        seen: set[str] = set()
        for u in urls:
            u = u.rstrip(".,;:!?")
            if u not in seen:
                seen.add(u)
                cleaned.append(u)

        header = f"_Research compiled on {today_human}. Recency window: last {config.NEWS_RECENCY_DAYS} days for news._\n\n"
        final_brief = header + brief.strip()

        return {"brief": final_brief, "source_urls": cleaned}

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


def run_research(agent: Any, topic: str) -> str:
    """Run the deep-research graph and return the consolidated research brief text."""
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
        "source_urls": [],
    }

    final_state: dict = {}
    try:
        for step in agent.stream(initial_state, config={"recursion_limit": 50}):
            for node_name, node_state in step.items():
                print(f"  - [graph] completed node: {node_name}", flush=True)
                final_state.update(node_state or {})
    except Exception as e:
        print(f"  - [warning] Research graph error: {e}", flush=True)

    return final_state.get("brief", "")


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


__all__ = ["build_research_agent", "run_research", "categorize_topic", "get_source_urls"]
