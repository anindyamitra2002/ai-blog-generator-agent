"""Writer chain: writes a single outlined section grounded in the research.

A fixed (non-agentic) step, run once per outlined section. Each call
produces a few hundred words of body prose that the editor will later
stitch together with intro, conclusion, and transitions.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

import config
from schemas import SectionOutline

WRITER_INSTRUCTIONS = """You are a skilled technical journalist and blog writer. Write a single section of a blog post as a comprehensive, detail-rich report.

Today's date is {today_human}. Write with the awareness that a reader is reading this on {today_human} — the post must feel current, not like a generic evergreen explainer.

Constraints:
- **Research brief is the ONLY source of truth**: Every fact, name, date, number, and claim you write MUST come from the research brief provided below. Do NOT supplement it with anything from your own training data or prior knowledge of this topic — even if you believe you know more about it. If your own knowledge and the research brief ever disagree, or the brief simply doesn't mention something you "recall", go with the brief (or omit it entirely) — never the reverse.
- **Fact-Dense Writing**: Ground every claim in the research brief provided. Include every concrete fact, statistic, percentage, date, metric, name, and specific example mentioned in the research brief for this section. Do NOT generalize, gloss over, or smooth out specific numbers/details.
- **Date Precision**: Whenever the research brief attaches a specific date to a fact or event, state that date explicitly in the prose (e.g. "On 3 July 2026, ..." or "As of {today_human}, ..."). Never substitute a vague word like "recently" or "currently" for a date that the research brief actually gives you.
- **Source Integration & Citations**: Explicitly integrate and cite the source websites, organizations, and precise URLs in the body text (e.g. 'According to a study by the [World Economic Forum](https://weforum.org),...' or 'as reported in the [Ministry of Finance press release](https://pib.gov.in)...'). Do NOT lean on the same one or two sources repeatedly — the research brief was compiled from many distinct sources specifically so different facts can be attributed to different outlets; draw from as many of the distinct sources present in the brief as are actually relevant to this section, not just the first ones mentioned. Always write the citation as a markdown link `[Source Title](URL)` using the real title given in the brief, not the bare URL on its own.
- **No Boilerplate Gaps, and No Invented Filler Either**: NEVER write meta-commentary, placeholder text, or filler like 'the research brief doesn't cover this' or 'there is a noted gap in the research brief regarding...'. Focus strictly on writing the factual, grounded content itself. If information is missing for a planned point, skip it entirely and expand on the details that *are* present in the research — do NOT invent a plausible-sounding fact, name, or number to fill the gap. A shorter, fully-grounded section is always better than a longer one that pads out with unverified content.
- **Accessible & Professional**: Write in a clear, conversational-but-professional tone, accessible to a general technical audience.
- **Length**: Target 300-500 words for this section to ensure all details and sources are fully reported.
- **Structure**: Use short, readable paragraphs (3-5 sentences).
- **Format**: Do not include the section heading in your output — only the body prose. Do not write a transition into the next section (the editor pass handles transitions). Use Markdown formatting only for emphasis (bold, italic) or short lists where it genuinely aids readability — do not add headers or sub-headers.

Return only the section body text, nothing else.
"""


def build_writer_chain(llm: BaseChatModel):
    """Build the writer prompt | llm chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_INSTRUCTIONS),
            (
                "human",
                "Section heading: {heading}\n"
                "Section intent: {intent}\n"
                "Key points to cover:\n{key_points}\n\n"
                "Research brief:\n{research}\n\n"
                "Write the section body.",
            ),
        ]
    ).partial(today_human=config.TODAY_HUMAN)
    return prompt | llm


def write_section(
    llm: BaseChatModel, section: SectionOutline, research_brief: str
) -> str:
    """Run the writer chain for one section and return its body text."""
    chain = build_writer_chain(llm)
    key_points = (
        "\n".join(f"- {p}" for p in section.key_points)
        if section.key_points
        else "(none specified)"
    )
    chunks = []
    for chunk in chain.stream(
        {
            "heading": section.heading,
            "intent": section.intent,
            "key_points": key_points,
            "research": research_brief,
        }
    ):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
        chunks.append(content)
    print("\n", flush=True)
    return "".join(chunks)


__all__ = ["build_writer_chain", "write_section"]