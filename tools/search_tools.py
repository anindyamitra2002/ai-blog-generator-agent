"""Instantiate LangChain's built-in research / search tools.

All tools here are maintained LangChain integrations — no custom search
client or scraper code. The research agent receives the entire list at
once and reasons about which tool to call and in what order, rather
than following a hardcoded call sequence.
"""

from __future__ import annotations

from config import (
    ENABLE_ARXIV,
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    SERPER_API_KEY,
)
from langchain_core.tools import BaseTool, tool

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


def modify_query_by_category(query: str, category: str) -> str:
    """Modifies the search query to focus strictly on the 10+ trusted domains for the category."""
    query = query.strip()
    sources = CATEGORY_SOURCES.get(category, CATEGORY_SOURCES["geopolitics_international_relations"])
    
    # Formulate site filter suffix: (site:domain1 OR site:domain2 OR ...)
    site_filter = " OR ".join(f"site:{domain}" for domain in sources)
    
    # Check if a site filter is already present in the query
    if "site:" in query.lower():
        return query
        
    return f"{query} ({site_filter})"


def build_search_tools(category: str = "geopolitics_international_relations") -> list[BaseTool]:
    """Return the list of research tools available to the agent, customized for the category.

    Tools included:
      1. Tavily search — AI-optimized, pre-ranked and summarized results (advanced depth, category filtered).
         (Skipped if TAVILY_API_KEY is not set.)
      2. Wikipedia query — canonical background facts, no API key required.
      3. DuckDuckGo search — general/fallback web search, no API key required (category filtered).
      4. Arxiv query — academic preprints, enabled by ENABLE_ARXIV.
      5. Google Serper search — deep search tool using Google SERP API (category filtered).
         (Skipped if SERPER_API_KEY is not set.)
    """
    tools: list[BaseTool] = []

    # --- 1. Tavily ---------------------------------------------------------
    if TAVILY_API_KEY:
        from langchain_tavily import TavilySearch

        sources = CATEGORY_SOURCES.get(category, CATEGORY_SOURCES["geopolitics_international_relations"])

        # Base tool with advanced search depth and native domain restrictions
        tavily_base = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=max(TAVILY_MAX_RESULTS, 10),
            search_depth="advanced",
            include_domains=sources,
        )

        @tool("tavily_search_deep")
        def tavily_search_deep(query: str) -> str:
            """A deep AI-optimized web search tool that returns pre-ranked, summarized
            results specifically from trusted, categorized resources. Input is a search query string.
            """
            return tavily_base.invoke({"query": query})

        tools.append(tavily_search_deep)

    # --- 2. Wikipedia ------------------------------------------------------
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wikipedia_base = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=1,
            doc_content_chars_max=1200,
        )
    )

    @tool("wikipedia")
    def wikipedia(query: str) -> str:
        """Useful for looking up canonical background facts, definitions,
        and historical context about well-known topics (people, places,
        concepts, events). Input should be a short search query — a topic
        name or a single noun phrase.
        """
        try:
            return wikipedia_base.invoke({"query": query})
        except Exception as e:
            return f"Wikipedia search is currently unavailable (Error: {e}). Please use other search tools like Tavily or DuckDuckGo."

    tools.append(wikipedia)

    # --- 3. DuckDuckGo -----------------------------------------------------
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    # Base tool localized to India English
    ddg_base = DuckDuckGoSearchResults(
        api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en"),
        num_results=max(TAVILY_MAX_RESULTS, 10),
    )

    @tool("duckduckgo_search_india")
    def duckduckgo_search_india(query: str) -> str:
        """A general-purpose web search engine localized to India. Use this for Indian news,
        regional references, blogs, and official press releases. Input is a search
        query string.
        """
        modified = modify_query_by_category(query, category)
        return ddg_base.invoke({"query": modified})

    tools.append(duckduckgo_search_india)

    # --- 4. (Optional) Arxiv ------------------------------------------------
    if ENABLE_ARXIV:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper

        arxiv = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(
                top_k_results=1,
                load_max_docs=1,
                doc_content_chars_max=1200,
            ),
            name="arxiv",
            description=(
                "Search academic preprints on arXiv. Use this when the topic "
                "would benefit from peer-reviewed scientific grounding — e.g. "
                "machine learning, physics, math, biology research. Input is "
                "a search query string."
            ),
        )
        tools.append(arxiv)

    # --- 5. Google Serper --------------------------------------------------
    if SERPER_API_KEY:
        from langchain_community.tools import GoogleSerperRun
        from langchain_community.utilities import GoogleSerperAPIWrapper

        # Base tool initialized with India geoloc and English language
        serper_base = GoogleSerperRun(
            api_wrapper=GoogleSerperAPIWrapper(
                serper_api_key=SERPER_API_KEY,
                gl="in",
                hl="en",
                k=10
            )
        )

        @tool("google_serper_search_deep")
        def google_serper_search_deep(query: str) -> str:
            """A deep Google search tool that retrieves organic search results and snippets
            specifically from Indian news portals, official publications, and press releases.
            Input is a search query string.
            """
            modified = modify_query_by_category(query, category)
            return serper_base.invoke({"query": modified})

        tools.append(google_serper_search_deep)

    return tools


__all__ = ["build_search_tools"]
