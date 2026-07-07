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
- **Fact-Dense Writing**: Ground every claim in the research brief provided. Include every concrete fact, statistic, percentage, date, metric, name, and specific example mentioned in the research brief for this section. Do NOT generalize, gloss over, or smooth out specific numbers/details.
- **Date Precision**: Whenever the research brief attaches a specific date to a fact or event, state that date explicitly in the prose (e.g. "On 3 July 2026, ..." or "As of {today_human}, ..."). Never substitute a vague word like "recently" or "currently" for a date that the research brief actually gives you.
- **Source Integration & Citations**: Explicitly integrate and cite the source websites, organizations, and precise URLs in the body text (e.g. 'According to a study by the [World Economic Forum](https://weforum.org),...' or 'as reported in the [Ministry of Finance press release](https://pib.gov.in)...'). Make sure to reference the specific sources provided in the research brief to build a highly authoritative, comprehensive report.
- **No Boilerplate Gaps**: NEVER write meta-commentary, placeholder text, or filler like 'the research brief doesn't cover this' or 'there is a noted gap in the research brief regarding...'. Focus strictly on writing the factual, grounded content itself. If information is missing for a planned point, skip it entirely and expand deeply on the details that *are* present in the research.
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
