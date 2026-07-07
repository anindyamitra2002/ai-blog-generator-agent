from __future__ import annotations

import re
from typing import Optional

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


def _try_unsplash(query: str) -> Optional[dict]:
    """Return the first qualifying Unsplash result, or None."""
    if not UNSPLASH_ACCESS_KEY:
        return None
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
        for photo in data.get("results", []):
            w = int(photo.get("width", 0))
            h = int(photo.get("height", 0))
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                user = photo.get("user", {}) or {}
                name = user.get("name", "Unknown")
                username = user.get("username", "")
                return {
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
                }
    except Exception:
        return None
    return None


def _try_pexels(query: str) -> Optional[dict]:
    """Return the first qualifying Pexels result, or None."""
    if not PEXELS_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 5, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for photo in data.get("photos", []):
            w = int(photo.get("width", 0))
            h = int(photo.get("height", 0))
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                photographer = photo.get("photographer", "Unknown")
                return {
                    "image_url": photo["src"]["large"],
                    "credit_text": f"Photo by {photographer} on Pexels",
                    "credit_url": photo.get("photographer_url") or "https://www.pexels.com",
                    "photographer": photographer,
                    "width": w,
                    "height": h,
                    "source": "pexels",
                }
    except Exception:
        return None
    return None


def _try_pixabay(query: str) -> Optional[dict]:
    """Return the first qualifying Pixabay result, or None."""
    if not PIXABAY_API_KEY:
        return None
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
        for photo in data.get("hits", []):
            w = int(photo.get("imageWidth", 0))
            h = int(photo.get("imageHeight", 0))
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                user = photo.get("user", "Unknown")
                return {
                    "image_url": photo["largeImageURL"],
                    "credit_text": f"Image by {user} on Pixabay",
                    "credit_url": "https://pixabay.com",
                    "photographer": user,
                    "width": w,
                    "height": h,
                    "source": "pixabay",
                }
    except Exception:
        return None
    return None


def _try_serper_image(query: str) -> Optional[dict]:
    """Try fetching cover image via Google Serper Images API."""
    if not SERPER_API_KEY:
        return None
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
        for img in data.get("images", []):
            w = int(img.get("imageWidth", 0))
            h = int(img.get("imageHeight", 0))
            # Ensure the image meets minimal width/height
            if w >= IMAGE_MIN_WIDTH and h >= IMAGE_MIN_HEIGHT:
                title = img.get("title", "")
                domain = img.get("domain", "Google Images via Serper")
                link = img.get("link", "")
                return {
                    "image_url": img["imageUrl"],
                    "credit_text": f"Image from {domain} ({title})" if title else f"Image from {domain}",
                    "credit_url": link or "https://serper.dev",
                    "photographer": domain,
                    "width": w,
                    "height": h,
                    "source": "serper",
                }
    except Exception as e:
        print(f"  - [warning] Serper image search failed: {e}", flush=True)
        return None
    return None


def _try_tavily_image(query: str) -> Optional[dict]:
    """Try fetching cover image via Tavily Search API."""
    if not TAVILY_API_KEY:
        return None
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
        if images:
            img_url = images[0]
            from urllib.parse import urlparse
            parsed = urlparse(img_url)
            domain = parsed.netloc or "Tavily Search"
            return {
                "image_url": img_url,
                "credit_text": f"Image from {domain} via Tavily Search",
                "credit_url": img_url,
                "photographer": domain,
                "width": IMAGE_MIN_WIDTH,
                "height": IMAGE_MIN_HEIGHT,
                "source": "tavily",
            }
    except Exception as e:
        print(f"  - [warning] Tavily image search failed: {e}", flush=True)
        return None
    return None


@tool("fetch_cover_image", args_schema=ImageQuery)
def fetch_cover_image(query: str) -> dict:
    """Fetch a landscape cover image for a blog post.

    Tries Serper and Tavily web image search first, falling back to Unsplash,
    Pexels, and Pixabay stock photo APIs. Returns the first valid image
    above the configured minimum resolution.

    Returns a dict with keys: image_url, credit_text, credit_url,
    photographer, width, height, source.
    """
    candidate_queries = _get_image_search_queries(query)
    
    # Try candidate queries in priority order across all providers
    for q in candidate_queries:
        for fn in (_try_serper_image, _try_tavily_image, _try_unsplash, _try_pexels, _try_pixabay):
            result = fn(q)
            if result:
                return result

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
