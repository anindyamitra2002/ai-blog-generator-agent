"""Build the tool-calling research agent.

The research phase is the only agentic stage in the pipeline — the model
decides which research tools to call and in what order, since research
needs vary by topic type. Everything downstream of research (outlining,
writing, editing) is a deterministic, fixed sequence.

This module uses ``langgraph.prebuilt.create_react_agent``, which is the
canonical agent API in LangChain 1.0+. The older ``AgentExecutor`` /
``create_tool_calling_agent`` imports from ``langchain.agents`` were
removed in LangChain 1.0 in favor of the LangGraph-based runtime.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from tools.search_tools import build_search_tools



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

RESEARCH_SYSTEM_PROMPT = """You are the research sub-agent of a blog-writing pipeline.

Your job: gather enough high-quality, sourced material on the user's topic \
so that downstream stages (outlining, writing, editing) can produce a \
well-grounded blog post.

CRITICAL CATEGORY-SPECIFIC DIRECTIVE:
{category_instructions}
Ensure that all search queries and facts you extract strictly prioritize these specified sources and domains to keep the post unbiased and authoritative.

A complete research pass should cover, where applicable:
1. A clear definition of the topic — what it is, why it matters.
2. Key subtopics or component concepts a reader needs to understand.
3. Recent developments, news, or notable examples from trusted sources.
4. At least 3-5 sourced facts, statistics, or specific examples with URLs from the specified domains.
5. Common misconceptions or contrasting viewpoints, if relevant.

You have access to several research tools (Tavily, Wikipedia, DuckDuckGo, \
Google Serper, and optionally Arxiv). Decide for yourself which tool(s) to call and in \
what order based on the topic. Different topics warrant different research \
sequences — do not call every tool mechanically, and do not call the same \
tool repeatedly with the same query.

CRITICAL EFFICIENCY GUIDELINES:
- Since you are running on a local CPU model, you must prioritize speed and conciseness.
- Do NOT make more than 2 tool calls in total. Usually, 1 Wikipedia query and 1 Tavily/DuckDuckGo/Serper Search query are more than enough.
- Do not search repeatedly. Once you have basic definitions and 2-3 links, proceed immediately to synthesize the research brief.

When you have enough material, write a single consolidated research brief \
with the following structure:

  ## Definition
  (short paragraph)

  ## Key subtopics
  - Subtopic A — one-line description
  - Subtopic B — one-line description
  - ...

  ## Recent developments / examples
  - Fact or example, with source URL in parentheses
  - ...

  ## Notable sources
  - URL 1
  - URL 2
  - ...

Do not write the blog post itself — only the research brief. End with the \
list of source URLs you cited.
"""


def categorize_topic(llm: BaseChatModel, topic: str) -> str:
    """Categorize the topic into one of the 21 categories."""
    prompt = (
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
        f"Topic: {topic}\n\n"
        "Response (category code name only):"
    )
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip().lower().replace("'", "").replace('"', '')
        for cat in CATEGORY_INSTRUCTIONS.keys():
            if cat in content:
                return cat
    except Exception as e:
        print(f"  - [warning] Categorization failed: {e}", flush=True)
    return "geopolitics_international_relations"


def build_research_agent(llm: BaseChatModel, category: str = "geopolitics_international_relations") -> Any:
    """Construct the tool-calling research agent.

    The agent is initialized with search tools customized for the given category.
    """
    tools = build_search_tools(category)

    # Attempt to load category-specific detailed prompt from file
    prompt_file = Path("prompts") / f"{category}.md"
    if prompt_file.exists():
        try:
            prompt = prompt_file.read_text(encoding="utf-8")
            print(f"  - Loaded detailed research prompt from: {prompt_file}", flush=True)
        except Exception as e:
            print(f"  - [warning] Failed to read prompt file {prompt_file}: {e}. Falling back to default system prompt.", flush=True)
            category_instructions = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["geopolitics_international_relations"])
            prompt = RESEARCH_SYSTEM_PROMPT.format(category_instructions=category_instructions)
    else:
        print(f"  - [warning] Detailed prompt file {prompt_file} not found. Falling back to default system prompt.", flush=True)
        category_instructions = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["geopolitics_international_relations"])
        prompt = RESEARCH_SYSTEM_PROMPT.format(category_instructions=category_instructions)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )
    return agent


def run_research(agent: Any, topic: str) -> str:
    """Run the research agent and return the consolidated research brief text.

    Uses `agent.stream` to output progress indicators in real-time,
    preventing the pipeline from feeling stuck on CPU.
    """
    print("  - Starting agentic research stream...", flush=True)
    
    last_message = None
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": f"Topic: {topic}"}]},
            config={"recursion_limit": 25},
        ):
            if "agent" in chunk:
                messages = chunk["agent"].get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tc in last_message.tool_calls:
                            print(f"  - Agent calling tool '{tc['name']}' with query: '{tc['args'].get('query', '')}'...", flush=True)
                    else:
                        print("  - Agent is compiling the research brief...", flush=True)
            elif "tools" in chunk:
                messages = chunk["tools"].get("messages", [])
                if messages:
                    for msg in messages:
                        content_len = len(msg.content) if hasattr(msg, "content") else 0
                        print(f"  - Tool '{msg.name}' returned {content_len} characters of results.", flush=True)
    except Exception as e:
        print(f"  - [warning] Research stream error: {e}", flush=True)

    if last_message is not None:
        if hasattr(last_message, "content"):
            return last_message.content
        if isinstance(last_message, dict):
            return last_message.get("content", "")
        return str(last_message)
    return ""


__all__ = ["build_research_agent", "run_research", "categorize_topic"]
