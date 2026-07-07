"""Outline chain: topic + research → structured Outline object.

This is a fixed (non-agentic) step. The raw research brief from the
research agent is passed to the LLM with a strict instruction to produce
a structured outline that downstream stages (writer, editor) can consume
deterministically.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from schemas import Outline, SectionOutline
import re


OUTLINE_INSTRUCTIONS = """You are a content strategist. Given a topic and a research brief, produce a structured outline for a blog post.

CRITICAL INSTRUCTIONS FOR DETAIL PRESERVATION:
- Your primary goal is to ensure the outline is a detailed map of the concrete facts, figures, statistics, names, and source references found in the research brief. Do NOT generalize, summarize vaguely, or smooth over details.
- Every section's key points MUST list specific, concrete data points, dates, metrics, case studies, or names directly from the research brief. 
- Map the 10+ sources cited in the research brief across the outline sections, specifying which source domains or URLs must be referenced in each section.
- Plan the sections strictly based on what is covered in the research brief. Do NOT plan sections or key points for which there is no research material, preventing the downstream writer from having to note gaps or invent boilerplate content.

The outline must include:
- A catchy, SEO-friendly title (not clickbait).
- A meta description of 1-2 sentences, ≤160 characters, summarizing the post for search engines.
- An ordered list of 4-7 sections, each with:
  - A clear heading.
  - A short intent statement describing what the section covers, its specific angle, and which research facts it is grounded in.
  - 2-4 highly specific key points (detailing facts, metrics, and source URLs) that the section must cover.

Do not include an "Introduction" or "Conclusion" section as separate items — the editor pass will add those. Aim for sections that flow naturally from one to the next.
"""


def try_parse_outline_markdown(text: str) -> Outline:
    """Robust fallback parser to extract an Outline object from markdown text."""
    lines = [line.strip() for line in text.split('\n')]
    
    title = ""
    meta_description = ""
    sections = []
    
    current_section = None
    collecting_key_points = False
    
    for idx, line in enumerate(lines):
        if not line:
            continue
            
        # Parse title
        if re.search(r'\*\*Title:?\*\*', line, re.IGNORECASE):
            match = re.search(r'\*\*Title:?\*\*\s*(.+)', line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
            elif idx + 1 < len(lines):
                title = lines[idx+1].strip()
            continue
            
        # Parse meta description
        if re.search(r'\*\*Meta\s*(?:description|Description):?\*\*', line, re.IGNORECASE):
            match = re.search(r'\*\*Meta\s*(?:description|Description):?\*\*\s*(.+)', line, re.IGNORECASE)
            if match:
                meta_description = match.group(1).strip()
            elif idx + 1 < len(lines):
                meta_description = lines[idx+1].strip()
            continue

        # Parse section header (e.g. "1. **The Core Components...**", "1. Inlet...")
        section_match = re.match(r'^(?:\d+[\.\)]|#+)\s*(?:\*\*)?([^\*\n]+?)(?:\*\*)?$', line)
        if not section_match:
            section_match = re.match(r'^(?:\d+[\.\)]|#+)\s*(?:\*\*)?([^\*\n\-\–\—]+)', line)
            
        if section_match and not re.search(r'intent:|key\s*points', line, re.IGNORECASE):
            heading = section_match.group(1).strip().strip(':').strip('**').strip()
            if heading and heading.lower() not in ["outline", "title", "meta description", "meta_description"]:
                if current_section:
                    sections.append(current_section)
                current_section = {"heading": heading, "intent": "", "key_points": []}
                collecting_key_points = False
                continue
                
        if current_section:
            # Parse intent
            intent_match = re.search(r'\*?Intent:?\*?\s*(.+)', line, re.IGNORECASE)
            if intent_match:
                current_section["intent"] = intent_match.group(1).strip()
                collecting_key_points = False
                continue
                
            # Check if key points section starts
            if re.search(r'key\s*points:?', line, re.IGNORECASE):
                collecting_key_points = True
                continue
                
            # Parse key point bullet
            bullet_match = re.match(r'^[\-\*\+\u2022]\s*(.+)', line)
            if bullet_match and collecting_key_points:
                current_section["key_points"].append(bullet_match.group(1).strip())
                
    if current_section:
        sections.append(current_section)
        
    if not title:
        title = "Untitled Blog Post"
    if not meta_description:
        meta_description = "Blog post about the selected topic."
        
    valid_sections = []
    for s in sections:
        if not s["heading"]:
            continue
        valid_sections.append(
            SectionOutline(
                heading=s["heading"],
                intent=s["intent"] or f"Discuss {s['heading']}.",
                key_points=s["key_points"] if s["key_points"] else ["General overview."]
            )
        )
        
    return Outline(title=title, meta_description=meta_description, sections=valid_sections)


def build_outline_chain(llm: BaseChatModel):
    """Build the prompt | llm | parser chain for outline generation."""
    parser = PydanticOutputParser(pydantic_object=Outline)

    # Attempt to bind JSON response format if supported
    if hasattr(llm, "bind"):
        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception:
            pass

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", OUTLINE_INSTRUCTIONS),
            ("system", "Respond strictly using this JSON format:\n{format_instructions}"),
            (
                "human",
                "Topic: {topic}\n\nResearch brief:\n{research}\n\nProduce the outline.",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def generate_outline(llm: BaseChatModel, topic: str, research_brief: str) -> Outline:
    """Run the outline chain and return a validated Outline object.

    Includes a robust markdown parser fallback to recover if the model
    returns markdown format instead of strictly structured JSON.
    """
    chain = build_outline_chain(llm)
    try:
        return chain.invoke({"topic": topic, "research": research_brief})
    except Exception as e:
        raw_text = getattr(e, "llm_output", None)
        if raw_text:
            try:
                parsed = try_parse_outline_markdown(raw_text)
                if parsed and parsed.sections:
                    print("  - [info] Successfully recovered outline using markdown fallback parser.", flush=True)
                    return parsed
            except Exception as parse_err:
                print(f"  - [warning] Fallback markdown parsing failed: {parse_err}", flush=True)
        raise e


__all__ = ["build_outline_chain", "generate_outline"]
