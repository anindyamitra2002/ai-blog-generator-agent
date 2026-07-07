from __future__ import annotations

import re

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import (
    IMAGE_MIN_HEIGHT,
    IMAGE_MIN_WIDTH,
    PEXELS_API_KEY,
    PIXABAY_API_KEY,
    UNSPLASH_ACCESS_KEY,
    TAVILY_API_KEY,
    SERPER_API_KEY,
)

# Many sites block/redirect requests that don't look like a real browser
# (no User-Agent, no Accept header) — instead of the actual image they
# serve a 403 HTML page, a tiny "hotlinking not allowed" placeholder, or a
# redirect to an unrelated page. If that response body gets saved to disk
# with a .jpg extension anyway, the result is a "corrupted" cover image
# that no viewer can open. These headers avoid the most common blocks, and
# _looks_like_image / _verify_image_url below make sure we never treat a
# non-image response as a valid image in the first place.
IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# Minimum plausible size (in bytes) for a real photo. Error pages, 1x1
# tracking pixels, and "hotlinking disabled" placeholders are almost always
# well under this, while any genuine landscape cover photo is well over it.
_MIN_IMAGE_BYTES = 2000

_IMAGE_MAGIC_BYTES = (
    b"\xff\xd8\xff",          # JPEG
    b"\x89PNG\r\n\x1a\n",     # PNG
    b"GIF87a",                # GIF
    b"GIF89a",                # GIF
    b"RIFF",                  # WEBP (RIFF....WEBP)
)


def _looks_like_image(content: bytes, content_type: str = "") -> bool:
    """Best-effort check that ``content`` is actually image data, not an
    HTML error page, JSON error body, or empty/placeholder response.
    """
    if not content or len(content) < _MIN_IMAGE_BYTES:
        return False
    if content_type and not content_type.lower().startswith("image/"):
        return False
    return any(content.startswith(magic) for magic in _IMAGE_MAGIC_BYTES)


def _verify_image_url(url: str) -> bool:
    """Actually download ``url`` and confirm it resolves to real, usable
    image bytes before we commit to it as the chosen cover image. This is
    what prevents a broken/blocked hotlink from ever being selected.
    """
    if not url:
        return False
    try:
        resp = requests.get(
            url, headers=IMAGE_REQUEST_HEADERS, timeout=15, allow_redirects=True
        )
        resp.raise_for_status()
        return _looks_like_image(resp.content, resp.headers.get("Content-Type", ""))
    except Exception:
        return False


class ImageQuery(BaseModel):
    """Input schema for the fetch_cover_image tool."""

    query: str = Field(
        ..., description="A descriptive search term for the cover image (e.g. 'machine learning neural network')."
    )


def _get_image_search_queries(title: str) -> list[str]:
    """Generate search queries for stock photo APIs to find relevant Indian content."""
    # Clean the title: remove quotes and common punctuation
    title = re.sub(r'["\'\.,\?!]', '', title).strip()
    
    # Split by colon, dash, or vertical bar to get the core topic
    parts = re.split(r'[:\-\—\|]', title)
    core = parts[0].strip() if parts else title
    
    queries = []
    
    # Candidate 1: Core topic with "India" (if it doesn't already have it)
    if "india" not in core.lower() and "indian" not in core.lower():
        queries.append(f"{core} India")
        queries.append(f"{core} Indian")
    else:
        queries.append(core)
        
    # Candidate 2: First 3-4 words of the core topic with "India" (if core is long)
    words = core.split()
    if len(words) > 3:
        short_core = " ".join(words[:3])
        if "india" not in short_core.lower() and "indian" not in short_core.lower():
            queries.append(f"{short_core} India")
        else:
            queries.append(short_core)
            
    # Candidate 3: Core topic as-is (fallback if localized search fails)
    if core not in queries:
        queries.append(core)
        
    # Candidate 4: A general keyword fallback if the core is too specific
    # Filter out common small/stop words for solo queries
    stop_words = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "by", "at", "between", "from"}
    for word in words:
        word_clean = word.strip().lower()
        if word_clean not in stop_words and len(word_clean) > 2:
            queries.append(word)
            queries.append(f"{word} India")
            
    # Filter out empty queries and remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_clean = q.strip()
        if q_clean and q_clean.lower() not in seen:
            seen.add(q_clean.lower())
            unique_queries.append(q_clean)
            
    return unique_queries


def _try_unsplash(query: str) -> list[dict]:
    """Return all qualifying Unsplash candidates, best first."""
    if not UNSPLASH_ACCESS_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": query,
                "per_page": 5,
                "orientation": "landscape",
                "content_filter": "high",
            },
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = []
        for photo in data.get("results", []):
            w = int(photo.get("width", 0))
            h = int(photo.get("height", 0))
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                user = photo.get("user", {}) or {}
                name = user.get("name", "Unknown")
                username = user.get("username", "")
                candidates.append({
                    "image_url": photo["urls"]["regular"],
                    "credit_text": f"Photo by {name} on Unsplash",
                    "credit_url": (
                        f"https://unsplash.com/@{username}"
                        if username else "https://unsplash.com"
                    ),
                    "photographer": name,
                    "width": w,
                    "height": h,
                    "source": "unsplash",
                })
        return candidates
    except Exception:
        return []


def _try_pexels(query: str) -> list[dict]:
    """Return all qualifying Pexels candidates, best first."""
    if not PEXELS_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 5, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = []
        for photo in data.get("photos", []):
            w = int(photo.get("width", 0))
            h = int(photo.get("height", 0))
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                photographer = photo.get("photographer", "Unknown")
                candidates.append({
                    "image_url": photo["src"]["large"],
                    "credit_text": f"Photo by {photographer} on Pexels",
                    "credit_url": photo.get("photographer_url") or "https://www.pexels.com",
                    "photographer": photographer,
                    "width": w,
                    "height": h,
                    "source": "pexels",
                })
        return candidates
    except Exception:
        return []


def _try_pixabay(query: str) -> list[dict]:
    """Return all qualifying Pixabay candidates, best first."""
    if not PIXABAY_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://pixabay.com/api/",
            params={
                "key": PIXABAY_API_KEY,
                "q": query,
                "per_page": 10,
                "image_type": "photo",
                "orientation": "horizontal",
                "min_width": IMAGE_MIN_WIDTH,
                "min_height": IMAGE_MIN_HEIGHT,
                "safesearch": "true",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        candidates = []
        for photo in data.get("hits", []):
            w = int(photo.get("imageWidth", 0))
            h = int(photo.get("imageHeight", 0))
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                user = photo.get("user", "Unknown")
                candidates.append({
                    "image_url": photo["largeImageURL"],
                    "credit_text": f"Image by {user} on Pixabay",
                    "credit_url": "https://pixabay.com",
                    "photographer": user,
                    "width": w,
                    "height": h,
                    "source": "pixabay",
                })
        return candidates
    except Exception:
        return []


def _try_serper_image(query: str) -> list[dict]:
    """Return all qualifying Serper Images candidates, best first."""
    if not SERPER_API_KEY:
        return []
    try:
        url = "https://google.serper.dev/images"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": 10
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        candidates = []
        for img in data.get("images", []):
            w = int(img.get("imageWidth", 0))
            h = int(img.get("imageHeight", 0))
            # Ensure the image meets minimal width/height
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                title = img.get("title", "")
                domain = img.get("domain", "Google Images via Serper")
                link = img.get("link", "")
                candidates.append({
                    "image_url": img["imageUrl"],
                    "credit_text": f"Image from {domain} ({title})" if title else f"Image from {domain}",
                    "credit_url": link or "https://serper.dev",
                    "photographer": domain,
                    "width": w,
                    "height": h,
                    "source": "serper",
                })
        return candidates
    except Exception as e:
        print(f"  - [warning] Serper image search failed: {e}", flush=True)
        return []


def _try_tavily_image(query: str) -> list[dict]:
    """Return all qualifying Tavily Search image candidates, best first."""
    if not TAVILY_API_KEY:
        return []
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "include_images": True,
            "max_results": 5
        }
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        images = data.get("images", [])
        candidates = []
        from urllib.parse import urlparse
        for img_url in images:
            parsed = urlparse(img_url)
            domain = parsed.netloc or "Tavily Search"
            candidates.append({
                "image_url": img_url,
                "credit_text": f"Image from {domain} via Tavily Search",
                "credit_url": img_url,
                "photographer": domain,
                # Tavily doesn't report dimensions; these are placeholders
                # confirming only that the download itself is validated
                # against the configured minimums, not actual pixel size.
                "width": IMAGE_MIN_WIDTH,
                "height": IMAGE_MIN_HEIGHT,
                "source": "tavily",
            })
        return candidates
    except Exception as e:
        print(f"  - [warning] Tavily image search failed: {e}", flush=True)
        return []


@tool("fetch_cover_image", args_schema=ImageQuery)
def fetch_cover_image(query: str) -> dict:
    """Fetch a landscape cover image for a blog post.

    Tries Serper and Tavily web image search first, falling back to Unsplash,
    Pexels, and Pixabay stock photo APIs. Every candidate returned by a
    provider is actually downloaded and checked to be real, sufficiently
    large image data before being selected, trying the next candidate (and
    then the next provider) on failure — this guards against broken or
    hotlink-blocked URLs (a common cause of a "corrupted" cover image) ever
    being returned.

    Returns a dict with keys: image_url, credit_text, credit_url,
    photographer, width, height, source.
    """
    candidate_queries = _get_image_search_queries(query)

    for q in candidate_queries:
        for fn in (_try_serper_image, _try_tavily_image, _try_unsplash, _try_pexels, _try_pixabay):
            for candidate in fn(q):
                if _verify_image_url(candidate["image_url"]):
                    return candidate

    # Graceful fallback — caller should still get a well-formed dict.
    return {
        "image_url": "",
        "credit_text": "No cover image available",
        "credit_url": "",
        "photographer": None,
        "width": 0,
        "height": 0,
        "source": "none",
    }


__all__ = ["fetch_cover_image", "ImageQuery"]