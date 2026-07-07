# AI Blog Generator Agent

A six-stage pipeline that turns a single topic string into a fully
researched, written, edited, illustrated, and assembled blog post — all
powered by LangChain's built-in tool integrations, a free local LLM
(Ollama), and free image APIs. Memory (self-hosted Mem0 + Chroma) lets
the agent recall or avoid duplicating past topics across sessions.

## Architecture

```
topic
  │
  ▼
[0] Memory recall  ──────────────► Mem0 (Chroma on disk)
  │
  ▼
[1] Research phase (agentic) ────► Tavily + Wikipedia + DuckDuckGo (+ Arxiv)
  │                                 LLM decides tool order
  ▼
[2] Outline generation (fixed) ──► Outline {title, meta_description, sections[]}
  │
  ▼
[3] Section writing (fixed) ─────► drafted body, one section at a time
  │
  ▼
[4] Editing pass (fixed) ────────► polished draft with intro + conclusion
  │
  ▼
[5] Image sourcing + assembly ───► Unsplash → Pexels → Pixabay, then post.md
  │
  ▼
[6] Memory update ───────────────► Mem0 records the completed run
```

The research phase is **agentic** (the model decides which tools to call
and in what order, since research needs vary by topic type). Everything
downstream of research — outlining, writing, editing — is a
**deterministic, fixed sequence**. This keeps output structure reliable
while still letting the research step adapt to the topic.

## Project layout

```
blog-agent/
├── .env                        # API keys and model config (not in VCS)
├── .env.example                # template — copy to .env and fill in
├── requirements.txt
├── main.py                     # CLI entrypoint, orchestrates the pipeline
├── config.py                   # loads .env, central settings
├── schemas.py                  # Pydantic models (Outline, SectionOutline, …)
├── assembler.py                # merges outline + body + image → post.md
├── tools/
│   ├── search_tools.py         # Tavily / Wikipedia / DuckDuckGo (+ Arxiv)
│   └── image_tool.py           # Unsplash → Pexels → Pixabay function tool
├── chains/
│   ├── outline_chain.py        # topic + research → structured Outline
│   ├── writer_chain.py         # outline section → drafted section
│   └── editor_chain.py         # full draft → polished draft
├── agent/
│   └── research_agent.py       # tool-calling research agent
├── memory/
│   └── mem0_store.py           # Mem0 + Chroma read/write helpers
├── outputs/                    # generated posts (one folder per slug)
└── memory_store/               # Mem0's local Chroma vector DB files
```

## Setup

### 1. Install Ollama and a tool-calling model

Install [Ollama](https://ollama.com/) (macOS / Linux / Windows), then
pull a tool-calling-capable model. Recommended:

```bash
ollama pull qwen2.5:7b
# or
ollama pull llama3.1:8b
```

Models below ~7B parameters frequently fail to invoke tools reliably —
avoid them for this project.

Verify tool-calling support **before** wiring anything together by
prompting the pulled model directly to invoke a hypothetical tool and
confirming it responds with a structured call rather than plain text.

### 2. Create a Python virtual environment

```bash
cd blog-agent
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

- `TAVILY_API_KEY` — get one free at <https://tavily.com>
  (~1,000 searches/month on the free tier).
- At least one image API key:
  - `UNSPLASH_ACCESS_KEY` — <https://unsplash.com/developers>
  - `PEXELS_API_KEY` — <https://www.pexels.com/api/>
  - `PIXABAY_API_KEY` — <https://pixabay.com/api/docs/>
- `OLLAMA_MODEL` — must match the model you pulled in step 1.

### 5. Verify each research tool standalone (recommended)

Before running the full pipeline, sanity-check each tool independently
to isolate any API-key or connectivity issues early:

```python
# In a Python shell inside the venv:
from tools.search_tools import build_search_tools
for t in build_search_tools():
    print(t.name, "→", t.invoke("large language models")[:200])
```

### 6. Verify the image tool standalone (recommended)

```python
from tools.image_tool import fetch_cover_image
print(fetch_cover_image.invoke({"query": "neural network"}))
```

## Run

```bash
python main.py "how retrieval-augmented generation (RAG) works"
```

Flags:

- `--skip-image` — skip the cover-image fetch (useful when no image API
  keys are configured yet).
- `--skip-memory` — skip the Mem0 recall/record steps (useful when
  `mem0ai` is not yet installed).

## Output

Every run produces a self-contained folder under `outputs/<slug>/`:

```
outputs/how-retrieval-augmented-generation-rag-works/
├── post.md          # final blog post (YAML frontmatter + body + sources)
├── meta.json        # run metadata (topic, slug, sources, paths, …)
└── assets/
    └── cover.jpg    # downloaded cover image (not hotlinked)
```

`post.md` shape:

```markdown
---
title: "..."
description: "..."
date: 2025-01-01
cover: assets/cover.jpg
image_credit: "Photo by Jane Doe on Unsplash"
---

![Cover image](assets/cover.jpg)

*Photo by Jane Doe on Unsplash*

# <Title>

<intro paragraph added by the editor>

## <Section heading>

<section body>

...

## Conclusion

<concluding paragraph>

## Sources

- https://example.com/source-1
- https://example.com/source-2
```

The YAML frontmatter is compatible with most static site generators
(Hugo, Jekyll, Astro, Eleventy, …).

## Validation checklist

Before considering the build "done":

- [ ] Chosen Ollama model confirmed to support tool calling
- [ ] Tavily tool returns results with the configured API key
- [ ] Wikipedia and DuckDuckGo tools work with zero configuration
- [ ] At least one image API key is active and returning valid image URLs
- [ ] A completed run produces a markdown file, an image, and a
      metadata file in the expected folder structure
- [ ] Mem0's local storage folder is created and populated after the
      first run
- [ ] A second, related topic correctly triggers a "similar topic
      found" signal from memory

## Optional extensions (not in v1)

- Set `ENABLE_ARXIV=true` in `.env` to add the Arxiv tool to the
  research agent's toolset — useful for academic / scientific topics.
- Swap the simple `create_tool_calling_agent` construction for
  LangGraph's state-graph agent pattern if resumable, checkpointed runs
  become necessary.
- Swap the self-hosted Mem0 store for Mem0's managed cloud tier if
  cross-machine memory persistence is needed later.
- Add an HTML rendering pass alongside the markdown output if a second
  output format is desired.

## Design notes

- **Why built-in tools only?** Tavily / Wikipedia / DuckDuckGo /
  Arxiv are all maintained LangChain integrations. The plan does not
  involve writing a custom search client or scraper.
- **Why is image sourcing a function tool, not part of the research
  agent?** There is no LangChain built-in for Unsplash / Pexels /
  Pixabay. The image tool is a single lightweight API-wrapper function
  (not a search engine) used by the assembler after the post title is
  known — it doesn't conflict with the "use built-in tools" requirement
  for the research/search layer.
- **Why is only the research phase agentic?** Research needs vary by
  topic type (a breaking-news topic needs different tool calls than a
  historical concept). Outlining, writing, and editing benefit from a
  deterministic, fixed sequence so output structure stays reliable.

Do we need the chains (outline, writer, editor). Because theu ar generalzing the content and removing the content specific details. Update these chains so that all the content specific details and make the blog a comprehensive report from all the 10+ sources.