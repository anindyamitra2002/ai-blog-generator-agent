"""Editor chain: full draft → polished draft with intro, conclusion, transitions.

A single fixed (non-agentic) pass over the concatenated draft. The editor
adds the intro and conclusion (which the writer deliberately omits),
smooths transitions between sections, and fixes tonal inconsistency —
without introducing new facts.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


EDITOR_INSTRUCTIONS = """You are a senior editor. Polish a complete draft blog post.

CRITICAL INSTRUCTIONS FOR DETAIL PRESERVATION:
- Your primary duty is to preserve every single concrete fact, statistic, percentage, date, metric, name, and source citation/URL present in the draft.
- Do NOT delete, generalize, or smooth out specific numbers, names, or reference links.
- Keep all inline citations (such as '[Source Name](URL)') intact. Do not remove markdown links.

Your job:
1. Add a strong introductory paragraph (2-4 sentences) at the very top, before the first section heading, that hooks the reader and previews the key facts and insights the post covers.
2. Add a concluding paragraph at the very end under a `## Conclusion` heading that synthesizes the key findings and source takeaways — do not just restate the intro.
3. Smooth the transitions between sections so the post reads as one continuous piece, not a list of standalone chunks, while retaining all details.
4. Fix any awkward phrasing, grammatical issues, or redundancy.
5. Preserve all factual claims and section headings — do not delete sections or rename headings.
6. Do not add new facts not present in the draft.

Return the complete polished post as Markdown in exactly this shape:

<intro paragraph — no heading>

## <First existing heading>

<first section body, possibly lightly revised but keeping all details and links>

## <Second existing heading>

<second section body>

...

## Conclusion

<concluding paragraph>
"""


def build_editor_chain(llm: BaseChatModel):
    """Build the editor prompt | llm chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EDITOR_INSTRUCTIONS),
            ("human", "Title: {title}\n\nDraft:\n{draft}\n\nReturn the polished post."),
        ]
    )
    return prompt | llm


def edit_draft(llm: BaseChatModel, title: str, draft: str) -> str:
    """Run the editor chain and return the polished markdown."""
    chain = build_editor_chain(llm)
    chunks = []
    for chunk in chain.stream({"title": title, "draft": draft}):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
        chunks.append(content)
    print("\n", flush=True)
    return "".join(chunks)


__all__ = ["build_editor_chain", "edit_draft"]
