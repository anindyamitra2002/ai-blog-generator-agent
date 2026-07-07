# AI Blog Generator Agent

An agentic, multi-stage blog-generation pipeline that turns a single topic query into a fully researched, detailed, fact-grounded, edited, and assembled blog post. It is powered by LangChain/LangGraph, the **Omniroute LLM Router** (or fallback local Ollama models), and multiple search/image APIs.

To ensure 100% factual accuracy and eliminate AI hallucinations, the agent uses an **agentic research loop** with parallel multi-source querying, strict grounding rules, and an automatic evidence-sufficiency safety net.

---

## 📖 For Non-Technical Users: How It Works Step-by-Step

Imagine you hire a team of human professionals to write a research-backed article. Here is how this AI agent replicates that human team:

### Step 1: Topic Categorization
First, the coordinator analyzes your topic and classifies it into one of **21 specialized categories** (such as *Artificial Intelligence*, *Finance & Economics*, or *Geopolitics*). This selection loads custom directives, specifying trusted websites (like `arXiv.org` for AI or `reuters.com` for finance) and how to handle research for that field.

### Step 2: The Deep-Research Agent (The Research Team)
Instead of searching once and guessing, the agent runs a **multi-turn research loop** managed by an agent coordinator:
* **The Planner** proposes 3-8 tailored search queries based on the category's trusted domains and today's actual date (so news is always up-to-date). It must include at least one query targeting recent news.
* **The Searchers** execute these queries in parallel using advanced search engines (Tavily, Serper, Exa, and Linkup). DuckDuckGo is used as a fallback if other keys aren't set.
* **The Synthesizer** collects all search results and writes a unified, factual **Research Brief**. It is bound by strict anti-hallucination rules: it cannot use its own pre-trained memory and must cite every URL next to every fact.
* **The Checker (Reflection)** reviews the brief to ensure it has enough unique sources (target: 10+) and includes recent, dated facts. If it is missing information, it specifies the gaps, and the Planner generates new, targeted follow-up queries. This loop repeats up to 3 times.
* **Safety Net (Evidence Sufficiency)**: If the search tools cannot find enough real evidence (e.g., less than 400 characters of genuine results), the agent stops and warns you instead of writing a made-up article. You can override this using the `--force-thin` flag.

### Step 3: Fact-Dense Outline Generation (The Editor-in-Chief)
Using the final research brief, the agent plans a detailed outline (4-7 sections). The outline maps out exactly which facts, dates, names, and source references from the brief go into each section. It also plans a "Latest Developments" section if recent news was found.

### Step 4: Section Writing (The Journalist)
A writer drafts the body of the article **one section at a time**. This stage writes detail-dense prose, explicitly including numbers, percentages, dates, and names. It integrates markdown links to the sources directly into the paragraph text (e.g., `According to a study by the [World Economic Forum](https://weforum.org)...`).

### Step 5: Senior Editing (The Copy Editor)
A senior editor reads the entire concatenated draft. The editor:
* Adds a compelling introductory paragraph situated in the current timeframe (referencing today's date).
* Adds a concluding paragraph that synthesizes the takeaways.
* Standardizes phrasing, fixes grammatical issues, and adds smooth transitions between sections.
* **Crucially**, the editor is forbidden from adding any new facts or removing any source links.

### Step 6: Image Sourcing & Assembly (The Publisher)
* The publisher uses the blog title to query image APIs (Unsplash, Pexels, or Pixabay) for a high-quality cover photo.
* It downloads the cover image locally so it won't break later due to hotlink blocks.
* It compiles the frontmatter (title, meta description, cover path, image credit) and the body into a standard markdown file (`post.md`) and writes a run metadata file (`meta.json`).

---

## 🛠️ Technical Architecture

This repository uses a hybrid architecture:
1. An **agentic, multi-turn state-graph loop** (using LangGraph) for the unpredictable, adaptive research phase.
2. A **deterministic, sequential chain pipeline** (using LangChain) for outlining, writing, editing, and compiling.

### System Workflow Diagram

```mermaid
graph TD
    Topic[User Topic] --> Categorize[Categorize Topic <br> 21 Categories]
    Categorize --> LoadDirective[Load prompts/research/category.md]
    
    %% LangGraph Research Loop
    subgraph LangGraph Deep-Research Loop
        LoadDirective --> InitState[Initialize ResearchState]
        InitState --> Plan[Plan Node: Propose 3-8 Queries]
        Plan --> Search[Search Node: Parallel Tool Execution]
        Search --> Synthesize[Synthesize Node: Compile Brief]
        Synthesize --> Reflect{Reflect Node: Check Recency & Sources}
        Reflect -- Gap Found & Iterations < Max --> Plan
        Reflect -- No Gaps / Iterations Max --> Finalize[Finalize Node: Structure Sources & Add Headers]
    end
    
    Finalize --> CheckEvidence{Evidence Sufficient? <br> > 400 chars}
    CheckEvidence -- No & --force-thin false --> Abort[Stop Pipeline: Print warning & thin brief]
    CheckEvidence -- Yes / --force-thin true --> Outline[Outline Chain: Pydantic Outline Generation]
    
    %% Sequential Pipeline
    Outline --> Writer[Writer Chain: Write Section-by-Section]
    Writer --> Editor[Editor Chain: Add Intro/Conclusion & Transitions]
    Editor --> Sourcing[Image Sourcing: Unsplash/Pexels/Pixabay]
    Sourcing --> Assemble[Assembleer: Download Image & Build post.md + meta.json]
    
    classDef loop fill:#f9f,stroke:#333,stroke-width:2px;
    class Plan,Search,Synthesize,Reflect,Finalize loop;
```

### Technical Components Details

#### 1. Primary LLM: Omniroute
By default, the pipeline runs on the **Omniroute LLM Router**, an OpenAI-compatible gateway that automatically routes LLM queries to the most optimal model based on prompt complexity and requirements.
* **Omniroute Base URL**: Local gateway `http://localhost:20128/v1` (or remote proxy `https://openrouter.ai/api/v1`).
* **Model Configuration**: Uses `model="auto"` which allows the Omniroute router to select the optimal model dynamically (e.g., `google/gemini-2.5-flash:free` or equivalent) for planning, writing, and editing.
* **Ollama (Optional Offline Local Model)**: Can be swapped in by changing `LLM_PROVIDER=ollama` in the `.env` configuration (relying on local model weights like `qwen2.5:7b` or `llama3.1:8b`).

**Temperature Strategy:**
* **Planner/Reflector (`PLANNER_TEMPERATURE = 0.1`)**: Low temperature ensures reliable, structured JSON query planning and gap checking.
* **Writer/Editor (`LLM_TEMPERATURE = 0.7`)**: Higher temperature allows for engaging, professional, and natural-sounding blog prose.

#### 2. Specialized Category Prompts & Generation
Category prompts live in [prompts/research/](file:///d:/Learning/ai-blog-generator-agent/prompts/research). They are generated dynamically from a single central template via [prompts/_generate_prompts.py](file:///d:/Learning/ai-blog-generator-agent/prompts/_generate_prompts.py). 

Each category markdown file contains:
* **CRITICAL CATEGORY-SPECIFIC DIRECTIVE**: Details the focus area and a curated list of trusted domain names for searches.
* **CRITICAL GROUNDING & ANTI-HALLUCINATION RULES**: Strict system instructions to prevent the model from leaning on pre-trained memory.
* **REDUNDANCY PREVENTION**: Guidelines to avoid structural and concept repetition.
* **OUTPUT FORMAT**: Specifies the exact markdown headers of the brief.

#### 3. Search Tool Priority & Fallbacks
Tools are loaded in [tools/search_tools.py](file:///d:/Learning/ai-blog-generator-agent/tools/search_tools.py). They are divided into:
* **Recency Restricted**: Filters search results to a specific timeframe (e.g., last 14 days). Tools include `tavily_news`, `serper_recent`, `exa_recent`, `linkup_recent`, `ddg_recent`.
* **Unrestricted Deep**: Searches for general background. Tools include `tavily_deep`, `serper_deep`, `exa_deep`, `linkup_deep`, `wikipedia`, `arxiv`, `ddg_deep`.

**Priority Mapping**:
The planning node dynamically resolves search queries. To maximize distinct sources:
1. It queries **Tavily, Serper, Exa, and Linkup** in parallel to gather diverse links and full pages.
2. **Arxiv** is queried only if the category is technical/scientific.
3. **DuckDuckGo** acts strictly as a **fallback of last resort**. If stronger search engine keys are available, DuckDuckGo is hidden from the planner's available tools to ensure maximum depth.

#### 4. Anti-Hallucination Guardrails
* **Evidence-Sufficiency Check**: `MIN_EVIDENCE_CHARS` (default 400) is checked against the total character count of genuine search results. If search fails to find info, the pipeline aborts rather than generating fake text.
* **Rigid Grounding Instructions**: Implemented across all chains:
  * [Outline Chain](file:///d:/Learning/ai-blog-generator-agent/chains/outline_chain.py) mapping facts directly from the brief.
  * [Writer Chain](file:///d:/Learning/ai-blog-generator-agent/chains/writer_chain.py) writing section bodies from outline items only.
  * [Editor Chain](file:///d:/Learning/ai-blog-generator-agent/chains/editor_chain.py) polishing text without adding external context.

#### 5. Link Formatting & Source Sourcing
* **Inline Link Formatting**: The writer is instructed to weave sources into the content using descriptive Markdown titles (`[Reuters: India GDP grows 7%](https://reuters.com/...)`) rather than raw URLs or generic text.
* **Sources Section**: The assembler parses both markdown links and raw URLs from the brief, performs title lookups against raw evidence, and outputs a clean, bulleted `## Sources` list at the bottom of the article.
* **Image Sourcing**: Fetches cover graphics via Unsplash, Pexels, or Pixabay, verifying and downloading image bytes to `assets/cover.jpg` (with hotlinking fallback only if local write fails).

---

## 📂 Project Structure

Here is the current, updated folder structure of the repository:

```
blog-agent/
├── .env                        # Local configurations & API keys (ignored by git)
├── .env.example                # Template configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Entrypoint; orchestrates categories & pipeline flow
├── config.py                   # Central settings loader and validation rules
├── schemas.py                  # Pydantic models (Outline, ImageResult, PostMetadata)
├── assembler.py                # Assembles frontmatter, downloads cover, writes post.md
├── test.py                     # Diagnostic script to test Omniroute connection and "auto" routing
├── test_headers.py             # Diagnostic script to check OpenRouter custom header integration
├── tools/
│   ├── search_tools.py         # Configures Tavily, Serper, Exa, Linkup, Arxiv, & DDG
│   └── image_tool.py           # Multi-provider image fetcher (Unsplash, Pexels, Pixabay)
├── chains/
│   ├── editor_chain.py         # Stitches body, adds intro/conclusion & transitions
│   ├── outline_chain.py        # Generates structured section plans from the brief
│   └── writer_chain.py         # Drafts sections individually with inline citations
├── agent/
│   ├── research_agent.py       # LangGraph deep-research StateGraph coordinator
│   └── state.py                # ResearchState definition for LangGraph
├── prompts/
│   ├── _generate_prompts.py    # Generates category-specific markdown files
│   └── research/               # Folder containing 21 category directive files (*.md)
└── outputs/                    # Output folder for generated articles (auto-created)
```

---

## 🚀 Setup & Installation (Omniroute Focused)

### 1. Create a Virtual Environment & Install Dependencies
Navigate to the project root directory and set up Python:
```bash
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and enter your API keys.

#### A. Configure the LLM (Omniroute Setup)
To use Omniroute as your LLM router, configure these keys under the OpenAI-compatible section:
```env
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:20128/v1  # Points to your local Omniroute router
OPENAI_API_KEY=your-omniroute-key-here    # Your Omniroute router API key
OPENAI_MODEL=auto                          # Keep as "auto" for dynamic smart routing
```

*(Optional Local Model Fallback)*: If you want to use local Ollama models instead:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b                    # Or llama3.1:8b
```

#### B. Configure Search API Keys
To get high-quality search results, fill in your search API keys:
```env
TAVILY_API_KEY=your-tavily-key
SERPER_API_KEY=your-serper-key
EXA_API_KEY=your-exa-key
LINKUP_API_KEY=your-linkup-key
```

#### C. Configure Image API Keys (At least one)
Add keys to fetch cover images automatically:
```env
UNSPLASH_ACCESS_KEY=your-unsplash-key
PEXELS_API_KEY=your-pexels-key
PIXABAY_API_KEY=your-pixabay-key
```

### 3. Verify Connections
You can run the diagnostic script to ensure your local Omniroute router is active and responding correctly:
```bash
python test.py
```
This sends a test request using the `auto` routing model and prints the response.

You can also run header diagnostics:
```bash
python test_headers.py
```

---

## 🏃 Running the Agent

To generate a blog post on a topic:
```bash
python main.py "How retrieval-augmented generation (RAG) works in enterprise search"
```

### Command Flags:
* `--skip-image`: Skips querying image APIs and downloading a cover.
* `--force-thin`: Forces the agent to continue writing the blog post even if search tools found very little evidence (bypassing the safety net).

---

## 📊 Run Output

Each execution creates a directory inside `outputs/<slug-name>/` containing:
1. `post.md`: The complete markdown article including YAML frontmatter, cover image markdown, the title, date/currency notes, structured sections, and a sources list.
2. `meta.json`: A structured JSON file recording the generation details (topic, sources used, output path).
3. `assets/cover.jpg`: The downloaded cover photo.