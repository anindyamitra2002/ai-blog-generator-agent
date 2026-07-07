"""Assemble the final blog post: markdown + frontmatter + cover image.

Given the polished body, title, description, image result, and a list of
structured sources (title + url — see ``schemas.Source``), this module
writes the finished markdown file to ``outputs/<slug>/post.md`` with YAML
frontmatter, downloads the cover image into ``outputs/<slug>/assets/``,
and writes a ``meta.json`` for debugging / resumability. Legacy callers
that pass a plain list of URL strings are still supported (see
``_coerce_sources``) — each bare URL gets its domain name as a fallback
title so the Sources section never renders a raw, unreadable link.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests

import config
from schemas import ImageResult, PostMetadata, Source


def slugify(text: str) -> str:
    """Turn a topic / title into a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


# Same rationale as tools/image_tool.py: a plain requests.get() with no
# browser-like headers gets blocked/redirected by a lot of hotlink-
# protected hosts, and the resulting HTML error page (or empty/placeholder
# body) was getting written straight to cover.jpg as if it were a real
# image — that's what "corrupted cover image" actually was.
_IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}
_MIN_IMAGE_BYTES = 2000
_IMAGE_MAGIC_BYTES = (
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"GIF87a",                # GIF
    b"GIF89a",                # GIF
    b"RIFF",                  # WEBP (RIFF....WEBP)
)


def _looks_like_image(content: bytes, content_type: str = "") -> bool:
    """Best-effort check that ``content`` is real image data, not an HTML
    error page, JSON error body, or empty/placeholder response."""
    if not content or len(content) < _MIN_IMAGE_BYTES:
        return False
    if content_type and not content_type.lower().startswith("image/"):
        return False
    return any(content.startswith(magic) for magic in _IMAGE_MAGIC_BYTES)


def download_image(url: str, target_path: Path) -> None:
    """Download an image to ``target_path``. No-op if URL is empty.

    Raises on network failure (caller falls back to hotlinking) AND on a
    response that downloads fine but isn't actually valid image data
    (caller should NOT hotlink in that case — the URL is genuinely broken,
    so hotlinking it would just show the reader the same broken image).
    """
    if not url:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=_IMAGE_REQUEST_HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    content = resp.content
    content_type = resp.headers.get("Content-Type", "")
    if not _looks_like_image(content, content_type):
        raise ValueError(
            f"response did not look like a valid image (content-type={content_type!r}, "
            f"{len(content)} bytes) — the source URL is likely hotlink-blocked or dead"
        )
    target_path.write_bytes(content)


def _yaml_escape(s: str) -> str:
    """Quote-and-escape a string for safe YAML frontmatter."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _coerce_sources(sources: list) -> list[Source]:
    """Accept either structured {title, url} dicts/Source objects (preferred)
    or legacy bare URL strings, and normalize to a list of ``Source``.

    Bare strings get the domain name as their title so the Sources section
    never renders a raw, unreadably-long URL as the link text.
    """
    from urllib.parse import urlparse

    out: list[Source] = []
    for s in sources:
        if isinstance(s, Source):
            out.append(s)
        elif isinstance(s, dict):
            url = s.get("url", "")
            title = s.get("title") or (urlparse(url).netloc or url)
            out.append(Source(title=title, url=url))
        else:
            url = str(s)
            title = urlparse(url).netloc or url
            out.append(Source(title=title, url=url))
    return out


def assemble_markdown(
    topic: str,
    title: str,
    description: str,
    body_markdown: str,
    image: ImageResult,
    sources: list,
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
        except ValueError as e:
            # The URL responded, but not with real image data (e.g. an
            # HTML "hotlinking disabled" page). Hotlinking the same URL
            # would show the reader that identical broken response, so
            # there's no point falling back to it — treat as no image.
            cover_image_rel = ""
            credit_line = f"No cover image available (source URL was invalid: {e})"
            print(f"  - [warning] Cover image download rejected: {e}", flush=True)
        except Exception as e:
            # A network/timeout/HTTP-status problem on our end, not proof
            # the URL itself is broken — hotlink as a best-effort fallback,
            # annotated so it's clear a local re-download wasn't confirmed.
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
    # Render each source as a proper markdown link — "- [Title](URL)" —
    # rather than a bare URL. A bare URL as link text is what was making
    # the rendered/preview post show long, unreadable raw links; wrapping
    # it as [Title](URL) shows a short readable title while the URL still
    # works as the link target.
    resolved_sources = _coerce_sources(sources)
    sources_block = ""
    if resolved_sources:
        lines = ["## Sources\n"]
        for s in resolved_sources:
            safe_title = s.title.replace("[", "(").replace("]", ")").strip() or s.url
            lines.append(f"- [{safe_title}]({s.url})")
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
        sources=resolved_sources,
        output_dir=str(output_dir),
    )
    (output_dir / "meta.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )

    return meta


__all__ = ["assemble_markdown", "slugify", "download_image"]