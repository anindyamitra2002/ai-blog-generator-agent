"""CLI entrypoint: orchestrates the full blog-generation pipeline.

Five stages, executed in order:
  1. Research phase      — agentic LangGraph deep-research loop: plan ->
                            search -> synthesize -> reflect (repeat until
                            the brief has dated, recent facts or the
                            iteration budget runs out) -> finalize. Uses a
                            per-category prompt file (prompts/research/*.md)
                            that forces the brief to be grounded ONLY in the
                            search evidence collected this run.
  2. Outline generation  — fixed step; topic + research → Outline object.
  3. Section writing     — fixed step; once per outlined section.
  4. Editing pass        — fixed step; full draft → polished draft.
  5. Image + assembly    — fetch cover image, write post.md + meta.json.

Usage:
    python main.py "the topic you want a blog post about"
    python main.py "some topic" --skip-image
    python main.py "some very niche/local topic" --force-thin
"""

from __future__ import annotations

import argparse
import sys
import io
import warnings

# Suppress deprecation and user warnings from third-party libraries (e.g., LangChain, LangGraph)
warnings.filterwarnings("ignore")

# Force stdout/stderr to use UTF-8 on Windows to prevent UnicodeEncodeError crashes
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import traceback
from pathlib import Path

# Ensure the project root is on sys.path so package imports work whether
# the script is run from inside blog-agent/ or from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from agent.research_agent import build_research_agent, run_research, categorize_topic  # noqa: E402
from chains.editor_chain import edit_draft  # noqa: E402
from chains.outline_chain import generate_outline  # noqa: E402
from chains.writer_chain import write_section  # noqa: E402
from assembler import assemble_markdown  # noqa: E402
from schemas import ImageResult  # noqa: E402
from tools.image_tool import fetch_cover_image  # noqa: E402


def print_stage(stage: str) -> None:
    print(f"\n{'=' * 3} {stage} {'=' * 3}")


def build_llm():
    """Construct the configured LLM (Ollama or OpenAI-compatible/Omniroute) from config."""
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add your Omniroute (or OpenAI/"
                "OpenRouter) API key to the OPENAI_API_KEY variable in your .env file."
            )

        # OPENAI_API_BASE points at the Omniroute router in this project's
        # default .env, and OPENAI_MODEL="auto" lets Omniroute pick the best
        # backing model per request — this is the "Omniroute LLM (auto model)"
        # the pipeline uses end-to-end (research planning/reflection, outline,
        # writing, and editing all share this one client).
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_REQUEST_TIMEOUT,
            default_headers={
                "HTTP-Referer": "https://github.com/diegosouzapw/blog-agent",
                "X-Title": "AI Blog Generator Agent",
            }
        )
    else:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_ctx=8192,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Blog Generator Agent — deep-research (date-aware, "
        "iterative LangGraph agent), outline, write, edit, and assemble a "
        "full, up-to-date blog post from a single topic string.",
    )
    p.add_argument("topic", help="Topic to generate a blog post about.")
    p.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip the cover image fetch (e.g. when no image API keys are configured).",
    )
    p.add_argument(
        "--force-thin",
        action="store_true",
        help=(
            "By default, if the search tools can't find substantive evidence for the "
            "topic, the pipeline stops after the research phase rather than risking a "
            "hallucinated post grounded in the LLM's own background knowledge. Pass "
            "this flag to continue anyway and generate the post from whatever thin "
            "evidence (or honest 'nothing found') the research phase produced."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    topic = args.topic.strip()
    if not topic:
        print("Error: topic must not be empty.", file=sys.stderr)
        return 2

    print(f"Run date: {config.TODAY_HUMAN} (recency window: last {config.NEWS_RECENCY_DAYS} days for news)")

    # --- Configuration validation -----------------------------------------
    issues = config.validate()
    for issue in issues:
        print(f"[warning] {issue}")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    llm = build_llm()

    # --- 1. Research phase (deep-research LangGraph) -----------------------
    print_stage("Research phase (deep search)")
    print(f"Topic: {topic}")

    print("  - Classifying topic category...", flush=True)
    category = categorize_topic(llm, topic)
    print(f"  - Classified Category: {category.upper()}", flush=True)

    research_graph = build_research_agent(llm, category)
    research_result = run_research(research_graph, topic)
    research_brief: str = research_result.get("brief", "")
    sources: list[dict] = research_result.get("sources", [])
    evidence_insufficient: bool = bool(research_result.get("evidence_insufficient", False))

    print(f"Research brief length: {len(research_brief)} chars")
    print(f"Extracted {len(sources)} distinct source(s) (target: {config.MIN_SOURCES_TARGET}+).")

    if evidence_insufficient:
        print(
            "\n[warning] The search tools could not find substantive evidence for "
            f"'{topic}'. Generating a post now risks the writer/editor stages leaning "
            "on the LLM's own (possibly outdated or wrong) background knowledge "
            "instead of verified sources.",
            file=sys.stderr,
        )
        if not args.force_thin:
            print(
                "Stopping here to avoid a hallucinated post. Re-run with --force-thin "
                "if you want to continue anyway with the limited/honest brief below, "
                "or try a more specific/differently-worded topic so search can find it.\n",
                file=sys.stderr,
            )
            print("--- Research brief so far ---\n")
            print(research_brief)
            return 3
        print("[info] --force-thin set — continuing with the limited evidence available.\n")

    # --- 2. Outline generation --------------------------------------------
    print_stage("Outline generation")
    outline = generate_outline(llm, topic, research_brief)
    print(f"Title: {outline.title}")
    print(f"Meta description: {outline.meta_description}")
    print(f"Sections ({len(outline.sections)}):")
    for s in outline.sections:
        print(f"  - {s.heading}")

    # --- 3. Section writing -----------------------------------------------
    print_stage("Section writing")
    drafted: list[str] = []
    total = len(outline.sections)
    for i, section in enumerate(outline.sections, 1):
        print(f"  [{i}/{total}] Writing: {section.heading}")
        body = write_section(llm, section, research_brief)
        drafted.append(f"## {section.heading}\n\n{body.strip()}")

    draft = "\n\n".join(drafted)
    print(f"Draft assembled: {len(draft)} chars across {total} section(s).")

    # --- 4. Editing pass --------------------------------------------------
    print_stage("Editing pass")
    polished = edit_draft(llm, outline.title, draft)
    print(f"Polished draft length: {len(polished)} chars")

    # --- 5. Image sourcing + assembly -------------------------------------
    print_stage("Image sourcing + assembly")
    image_result = ImageResult(
        image_url="",
        credit_text="No cover image available",
        credit_url="",
        photographer=None,
        width=0,
        height=0,
        source="none",
    )

    if not args.skip_image:
        try:
            raw = fetch_cover_image.invoke({"query": outline.title})
            image_result = ImageResult(**raw)
            if image_result.source != "none":
                print(
                    f"Image source: {image_result.source} "
                    f"({image_result.width}x{image_result.height})"
                )
            else:
                print("[warning] No image returned by any provider.")
        except Exception as e:
            print(f"[warning] Image fetch failed: {e}")
    else:
        print("[info] Image step skipped (--skip-image).")

    meta = assemble_markdown(
        topic=topic,
        title=outline.title,
        description=outline.meta_description,
        body_markdown=polished,
        image=image_result,
        sources=sources,
        outputs_root=config.OUTPUTS_DIR,
    )

    post_path = Path(meta.output_dir) / "post.md"
    meta_path = Path(meta.output_dir) / "meta.json"
    print(f"\nPost written to: {post_path}")
    print(f"Metadata:        {meta_path}")
    if meta.cover_image_path:
        print(f"Cover image:     {Path(meta.output_dir) / meta.cover_image_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)