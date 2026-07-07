"""Structured data definitions for the blog-agent pipeline.

These Pydantic models are used both for runtime validation and as the
schema for the outline chain's structured output, so the LLM can be
constrained to emit exactly the fields the downstream stages expect.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SectionOutline(BaseModel):
    """A single planned section of the blog post."""

    heading: str = Field(
        ..., description="Section heading as it will appear in the post."
    )
    intent: str = Field(
        ...,
        description="What this section should cover and the angle it should take.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet points the section must touch on (2-4 items).",
    )


class Outline(BaseModel):
    """The full planned structure of a blog post."""

    title: str = Field(..., description="Catchy, SEO-friendly post title.")
    meta_description: str = Field(
        ...,
        description="1-2 sentence meta description for SEO, ≤160 characters.",
    )
    sections: list[SectionOutline] = Field(
        ..., description="Ordered list of sections that compose the post body."
    )


class DraftedSection(BaseModel):
    """A section that has been written but not yet edited."""

    heading: str
    body: str


class ImageResult(BaseModel):
    """The result of an image-sourcing call."""

    image_url: str = Field(
        ..., description="Direct URL to the chosen image, or empty if none found."
    )
    credit_text: str = Field(
        ...,
        description=(
            "Human-readable attribution line, e.g. 'Photo by Jane Doe on Unsplash'. "
            "Must be displayed alongside the image per the API's terms."
        ),
    )
    credit_url: str = Field(..., description="URL to the photographer's profile / source site.")
    photographer: Optional[str] = Field(None, description="Photographer's display name, if known.")
    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    source: Literal["unsplash", "pexels", "pixabay", "serper", "tavily", "none"] = Field(
        ..., description="Which API provided the image, or 'none' if all failed."
    )


class PostMetadata(BaseModel):
    """Run metadata written to meta.json for debugging and resumability."""

    topic: str
    slug: str
    title: str
    description: str
    cover_image_path: str
    cover_image_credit: str
    sources: list[str] = Field(default_factory=list)
    output_dir: str


__all__ = [
    "SectionOutline",
    "Outline",
    "DraftedSection",
    "ImageResult",
    "PostMetadata",
]
