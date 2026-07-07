"""Assemble the final blog post: markdown + frontmatter + cover image.

Given the polished body, title, description, image result, and source
list, this module writes the finished markdown file to
``outputs/<slug>/post.md`` with YAML frontmatter, downloads the cover
image into ``outputs/<slug>/assets/``, and writes a ``meta.json`` for
debugging / resumability.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

import config
from schemas import ImageResult, PostMetadata


def slugify(text: str) -> str:
    """Turn a topic / title into a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


def download_image(url: str, target_path: Path) -> None:
    """Download an image to ``target_path``. No-op if URL is empty."""
    if not url:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    target_path.write_bytes(resp.content)


def _yaml_escape(s: str) -> str:
    """Quote-and-escape a string for safe YAML frontmatter."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def assemble_markdown(
    topic: str,
    title: str,
    description: str,
    body_markdown: str,
    image: ImageResult,
    sources: list[str],
    outputs_root: Path,
) -> PostMetadata:
    """Build the final markdown file with frontmatter + image + sources.

    Returns a ``PostMetadata`` object describing the written run. Also
    writes ``meta.json`` next to ``post.md`` for debugging / resumability.
    """
    slug = slugify(title)
    output_dir = outputs_root / slug
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # --- Cover image ------------------------------------------------------
    image_filename = "cover.jpg"
    image_path = assets_dir / image_filename

    if image.image_url:
        try:
            download_image(image.image_url, image_path)
            cover_image_rel = f"assets/{image_filename}"
            credit_line = image.credit_text
        except Exception as e:
            # Fallback: hotlink + annotate the failure.
            cover_image_rel = image.image_url
            credit_line = f"{image.credit_text} (local download failed: {e})"
    else:
        cover_image_rel = ""
        credit_line = "No cover image available."

    date_str = config.TODAY_STR

    # --- YAML frontmatter -------------------------------------------------
    frontmatter = (
        "---\n"
        f'title: "{_yaml_escape(title)}"\n'
        f'description: "{_yaml_escape(description)}"\n'
        f"date: {date_str}\n"
        f"last_updated: {date_str}\n"
        f"cover: {cover_image_rel}\n"
        f'image_credit: "{_yaml_escape(credit_line)}"\n'
        "---\n\n"
    )

    # --- Cover image block ------------------------------------------------
    image_block = ""
    if cover_image_rel:
        image_block = (
            f"![Cover image]({cover_image_rel})\n\n"
            f"*{credit_line}*\n\n"
        )

    # --- Currency note ------------------------------------------------------
    currency_note = f"*Published {config.TODAY_HUMAN}. Data current as of this date.*\n\n"

    # --- Sources block ----------------------------------------------------
    sources_block = ""
    if sources:
        lines = ["## Sources\n"]
        for s in sources:
            lines.append(f"- {s}")
        sources_block = "\n".join(lines) + "\n"

    # --- Title + body -----------------------------------------------------
    # The editor pass already produced a clean intro paragraph + section
    # headings + conclusion, so we just prepend the H1 title and a
    # currency note before it.
    full_markdown = (
        frontmatter
        + image_block
        + f"# {title}\n\n"
        + currency_note
        + body_markdown.strip()
        + "\n\n"
        + sources_block
    )

    md_path = output_dir / "post.md"
    md_path.write_text(full_markdown, encoding="utf-8")

    # --- Metadata file ----------------------------------------------------
    meta = PostMetadata(
        topic=topic,
        slug=slug,
        title=title,
        description=description,
        generated_date=date_str,
        cover_image_path=cover_image_rel,
        cover_image_credit=credit_line,
        sources=sources,
        output_dir=str(output_dir),
    )
    (output_dir / "meta.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )

    return meta


__all__ = ["assemble_markdown", "slugify", "download_image"]
