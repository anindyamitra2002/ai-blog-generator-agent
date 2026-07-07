"""Instantiate LangChain's built-in research / search tools.

All tools here are maintained LangChain integrations — no custom search
client or scraper code. Two flavors of most engines are provided:

  * a "recent" / "news" variant that is date-window restricted (Tavily's
    native `topic="news"` + `days=`, Serper's `tbs=qdr:*` time filter, and
    DuckDuckGo's `time=` window) — used by the deep-research graph whenever
    it needs to answer "what happened recently".
  * a "deep" / general variant with no time restriction — used for
    background, definitions, and canonical context.

The deep-research graph (``agent/research_agent.py``) decides for itself,
per iteration, which of these tools to call and with what query, rather
than following a hardcoded call sequence.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from config import (
    ENABLE_ARXIV,
    NEWS_RECENCY_DAYS,
    SERPER_API_KEY,
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
)

CATEGORY_SOURCES = {
    "geopolitics_international_relations": [
        "abplive.com", "ndtv.com", "aninews.in", "pib.gov.in", "mea.gov.in",
        "foreignaffairs.com", "stimson.org", "clingendael.org", "orfonline.org", "idsa.in"
    ],
    "national_news_politics": [
        "abplive.com", "ndtv.com", "aninews.in", "tribuneindia.com", "indianexpress.com",
        "thehindu.com", "timesofindia.indiatimes.com", "deccanherald.com", "pib.gov.in", "ddnews.gov.in"
    ],
    "artificial_intelligence_ml": [
        "arxiv.org", "paperswithcode.com", "openai.com", "deepmind.google", "huggingface.co",
        "keras.io", "pytorch.org", "github.com", "distill.pub", "towardsdatascience.com"
    ],
    "software_development_programming": [
        "stackoverflow.com", "dev.to", "github.com", "hashnode.dev", "news.ycombinator.com",
        "freecodecamp.org", "w3schools.com", "developer.mozilla.org", "geeksforgeeks.org", "medium.com"
    ],
    "finance_investing_economics": [
        "economictimes.indiatimes.com", "livemint.com", "business-standard.com", "moneycontrol.com",
        "rbi.org.in", "finmin.nic.in", "bloomberg.com", "reuters.com", "investopedia.com", "ft.com"
    ],
    "health_medicine_healthcare": [
        "who.int", "ncbi.nlm.nih.gov", "mohfw.gov.in", "icmr.nic.in", "thelancet.com",
        "nejm.org", "cdc.gov", "mayoclinic.org", "healthline.com", "nature.com"
    ],
    "space_exploration_astronomy": [
        "isro.gov.in", "nasa.gov", "esa.int", "space.com", "sciencedaily.com",
        "astronomy.com", "skyandtelescope.org", "planetary.org", "arxiv.org", "nature.com"
    ],
    "climate_change_environment": [
        "ipcc.ch", "unep.org", "moef.gov.in", "wri.org", "sciencedirect.com",
        "worldwildlife.org", "greenpeace.org", "downtoearth.org.in", "cise.in", "nature.com"
    ],
    "legal_constitution_policy": [
        "sci.gov.in", "indiankanoon.org", "legislative.gov.in", "prsindia.org", "barandbench.com",
        "livelaw.in", "lawyersclubindia.com", "mha.gov.in", "scconline.com", "taxmann.com"
    ],
    "education_academia": [
        "education.gov.in", "ugc.ac.in", "aicte-india.org", "nptel.ac.in", "coursera.org",
        "edx.org", "khanacademy.org", "scholar.google.com", "researchgate.net", "academia.edu"
    ],
    "history_archaeology": [
        "asi.nic.in", "history.com", "britannica.com", "nationalgeographic.com", "ancient.eu",
        "jstor.org", "archeology.org", "unesco.org", "archive.org", "bl.uk"
    ],
    "cryptocurrency_blockchain": [
        "coindesk.com", "cointelegraph.com", "ethereum.org", "bitcoin.org", "vitalik.ca",
        "arxiv.org", "messari.io", "github.com", "blockworks.co", "coinmarketcap.com"
    ],
    "business_management_entrepreneurship": [
        "hbr.org", "forbes.com", "inc.com", "entrepreneur.com", "techcrunch.com",
        "startupindia.gov.in", "yourstory.com", "trak.in", "fastcompany.com", "ycombinator.com"
    ],
    "cybersecurity_privacy": [
        "cert-in.org.in", "owasp.org", "cisa.gov", "sans.org", "portswigger.net",
        "bleepingcomputer.com", "thehackernews.com", "darkreading.com", "schneier.com", "wired.com"
    ],
    "automobile_transport": [
        "autocarindia.com", "carandbike.com", "zigwheels.com", "morth.nic.in", "customs.gov.in",
        "electrek.co", "motor1.com", "sae.org", "team-bhp.com", "overdrive.in"
    ],
    "gadgets_consumer_electronics": [
        "gadgets360.com", "techradar.com", "theverge.com", "engadget.com", "cnet.com",
        "gsmarena.com", "digitaltrends.com", "wired.com", "tomshardware.com", "gizmodo.com"
    ],
    "agriculture_food_tech": [
        "icar.org.in", "agricoop.nic.in", "fao.org", "agriland.ie", "agriculture.com",
        "sciencedirect.com", "foodnavigator.com", "ift.org", "apeda.gov.in", "krishijagran.com"
    ],
    "sports_athletics": [
        "espncricinfo.com", "sports.ndtv.com", "sportskeeda.com", "bcci.tv", "olympics.com",
        "fifa.com", "abplive.com", "espn.com", "theathletic.com", "sportstar.thehindu.com"
    ],
    "entertainment_media_pop_culture": [
        "imdb.com", "rottentomatoes.com", "variety.com", "hollywoodreporter.com", "ndtv.com",
        "pinkvilla.com", "filmcompanion.in", "abplive.com", "billboard.com", "rollingstone.com"
    ],
    "travel_tourism_hospitality": [
        "incredibleindia.org", "lonelyplanet.com", "tripadvisor.com", "nationalgeographic.com", "cntraveller.in",
        "travelandleisure.com", "skift.com", "nomadicmatt.com", "fodorstravel.com", "roughguides.com"
    ],
    "sociology_anthropology_social_issues": [
        "un.org", "undp.org", "oxfam.org", "niti.gov.in", "wango.org",
        "pewresearch.org", "epw.in", "worldbank.org", "humanrightswatch.org", "amnesty.org"
    ]
}

DEFAULT_CATEGORY = "geopolitics_international_relations"


def modify_query_by_category(query: str, category: str) -> str:
    """Restrict a query to the trusted domain list for its category."""
    query = query.strip()
    sources = CATEGORY_SOURCES.get(category, CATEGORY_SOURCES[DEFAULT_CATEGORY])
    site_filter = " OR ".join(f"site:{domain}" for domain in sources)
    if "site:" in query.lower():
        return query
    return f"{query} ({site_filter})"


def build_search_toolkit(category: str = DEFAULT_CATEGORY) -> dict[str, BaseTool]:
    """Return a dict mapping logical tool keys to ready-to-invoke tools.

    Keys match ``schemas.PlannedQuery.tool`` so the research graph can look
    up ``toolkit[planned.tool].invoke({"query": planned.query})`` directly,
    without going through an agentic tool-selection loop.

    Keys: tavily_news, tavily_deep, serper_recent, serper_deep, wikipedia,
    ddg_recent, ddg_deep, arxiv (only if ENABLE_ARXIV / keys present).
    """
    sources = CATEGORY_SOURCES.get(category, CATEGORY_SOURCES[DEFAULT_CATEGORY])
    toolkit: dict[str, BaseTool] = {}

    # --- Tavily: recent news + general deep search --------------------
    if TAVILY_API_KEY:
        from langchain_tavily import TavilySearch

        tavily_news_base = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=max(TAVILY_MAX_RESULTS, 8),
            search_depth="advanced",
            topic="news",
            days=NEWS_RECENCY_DAYS,
            include_domains=sources,
        )

        @tool("tavily_news")
        def tavily_news(query: str) -> str:
            """Search recent NEWS only (last few days/weeks) via Tavily's
            news index, restricted to trusted category domains. Use this for
            'latest developments', 'this week', or anything time-sensitive.
            Input is a search query string — include the topic and,
            ideally, the word 'latest' or a specific recent timeframe.
            """
            try:
                return str(tavily_news_base.invoke({"query": query}))
            except Exception as e:
                return f"Tavily news search failed: {e}"

        toolkit["tavily_news"] = tavily_news

        tavily_deep_base = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=max(TAVILY_MAX_RESULTS, 8),
            search_depth="advanced",
            topic="general",
            include_domains=sources,
        )

        @tool("tavily_deep")
        def tavily_deep(query: str) -> str:
            """Deep, AI-ranked general web search via Tavily, restricted to
            trusted category domains. Use for background, definitions, and
            established context (not time-sensitive). Input is a query string.
            """
            try:
                return str(tavily_deep_base.invoke({"query": query}))
            except Exception as e:
                return f"Tavily deep search failed: {e}"

        toolkit["tavily_deep"] = tavily_deep

    # --- Wikipedia -------------------------------------------------------
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wikipedia_base = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
    )

    @tool("wikipedia")
    def wikipedia(query: str) -> str:
        """Canonical background facts, definitions, and historical context
        for well-known people, places, concepts, and events. NOT for recent
        news. Input should be a short topic name or noun phrase.
        """
        try:
            return wikipedia_base.invoke({"query": query})
        except Exception as e:
            return f"Wikipedia search unavailable ({e}). Use another tool instead."

    toolkit["wikipedia"] = wikipedia

    # --- DuckDuckGo: recent + general -------------------------------------
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    ddg_recent_base = DuckDuckGoSearchResults(
        api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en", time="w"),
        num_results=max(TAVILY_MAX_RESULTS, 8),
    )

    @tool("ddg_recent")
    def ddg_recent(query: str) -> str:
        """General-purpose web search restricted to results from the past
        week, localized to India. Use for recent news when Tavily/Serper
        are unavailable. Input is a search query string.
        """
        modified = modify_query_by_category(query, category)
        try:
            return ddg_recent_base.invoke({"query": modified})
        except Exception as e:
            return f"DuckDuckGo recent search failed: {e}"

    toolkit["ddg_recent"] = ddg_recent

    ddg_deep_base = DuckDuckGoSearchResults(
        api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en"),
        num_results=max(TAVILY_MAX_RESULTS, 8),
    )

    @tool("ddg_deep")
    def ddg_deep(query: str) -> str:
        """General-purpose web search localized to India, no time
        restriction. Use for background, official releases, regional
        references. Input is a search query string.
        """
        modified = modify_query_by_category(query, category)
        try:
            return ddg_deep_base.invoke({"query": modified})
        except Exception as e:
            return f"DuckDuckGo search failed: {e}"

    toolkit["ddg_deep"] = ddg_deep

    # --- (Optional) Arxiv --------------------------------------------------
    if ENABLE_ARXIV:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper

        arxiv_base = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(
                top_k_results=3, load_max_docs=3, doc_content_chars_max=1500
            )
        )

        @tool("arxiv")
        def arxiv(query: str) -> str:
            """Search academic preprints on arXiv. Use for machine learning,
            physics, math, or biology research topics. Input is a query string.
            """
            try:
                return arxiv_base.invoke({"query": query})
            except Exception as e:
                return f"Arxiv search failed: {e}"

        toolkit["arxiv"] = arxiv

    # --- Google Serper: recent (tbs time filter) + general ------------------
    if SERPER_API_KEY:
        from langchain_community.utilities import GoogleSerperAPIWrapper

        # Map the recency window to Google's "tbs" time-filter buckets.
        if NEWS_RECENCY_DAYS <= 1:
            tbs_value = "qdr:d"
        elif NEWS_RECENCY_DAYS <= 7:
            tbs_value = "qdr:w"
        elif NEWS_RECENCY_DAYS <= 31:
            tbs_value = "qdr:m"
        else:
            tbs_value = "qdr:y"

        try:
            serper_recent_wrapper = GoogleSerperAPIWrapper(
                serper_api_key=SERPER_API_KEY, gl="in", hl="en", k=10, tbs=tbs_value, type="news"
            )
        except TypeError:
            # Older langchain_community versions may not accept tbs/type.
            serper_recent_wrapper = GoogleSerperAPIWrapper(
                serper_api_key=SERPER_API_KEY, gl="in", hl="en", k=10
            )

        @tool("serper_recent")
        def serper_recent(query: str) -> str:
            """Google News search via Serper, time-filtered to the recent
            window. Use this specifically to find the latest news, events,
            or announcements. Input is a search query string.
            """
            modified = modify_query_by_category(query, category)
            try:
                return serper_recent_wrapper.run(modified)
            except Exception as e:
                return f"Serper recent search failed: {e}"

        toolkit["serper_recent"] = serper_recent

        serper_deep_wrapper = GoogleSerperAPIWrapper(
            serper_api_key=SERPER_API_KEY, gl="in", hl="en", k=10
        )

        @tool("serper_deep")
        def serper_deep(query: str) -> str:
            """General Google search via Serper (organic results), no time
            restriction. Use for background, official publications, and
            established context. Input is a search query string.
            """
            modified = modify_query_by_category(query, category)
            try:
                return serper_deep_wrapper.run(modified)
            except Exception as e:
                return f"Serper search failed: {e}"

        toolkit["serper_deep"] = serper_deep

    return toolkit


def available_tool_keys(toolkit: dict[str, BaseTool]) -> list[str]:
    """List of tool keys actually available given current API key config."""
    return list(toolkit.keys())


__all__ = [
    "CATEGORY_SOURCES",
    "modify_query_by_category",
    "build_search_toolkit",
    "available_tool_keys",
]
