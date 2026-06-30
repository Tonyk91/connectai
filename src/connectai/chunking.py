"""Markdown article loading and semantic, token-aware chunking.

Chunks aim for ~300-500 tokens with a small overlap so that a single support
procedure is rarely split across a boundary. Token counts are approximated from
word counts (≈1.3 tokens/word) to avoid a tokenizer dependency at ingest time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKENS_PER_WORD = 1.3
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Article:
    """A single knowledge-base support article."""

    id: str
    title: str
    category: str
    body: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of an article."""

    id: str
    article_id: str
    title: str
    category: str
    text: str
    ordinal: int


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text`` from its word count."""
    words = len(text.split())
    return int(round(words * _TOKENS_PER_WORD))


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split a ``---`` YAML-ish frontmatter block from the markdown body.

    Only flat ``key: value`` pairs are supported, which is all the KB needs.
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw.strip()

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    body = raw[match.end() :].strip()
    return meta, body


def article_from_markdown(raw: str, fallback_id: str) -> Article:
    """Build an :class:`Article` from raw markdown with frontmatter."""
    meta, body = parse_frontmatter(raw)
    return Article(
        id=meta.get("id", fallback_id),
        title=meta.get("title", fallback_id),
        category=meta.get("category", "general"),
        body=body,
    )


def _split_paragraphs(body: str) -> list[str]:
    # Drop markdown headings markers but keep their text as part of the following block.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return paragraphs


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def chunk_article(article: Article, target_tokens: int = 400, overlap_tokens: int = 60) -> list[Chunk]:
    """Chunk one article into ~``target_tokens`` pieces with sentence overlap."""
    units = _split_paragraphs(article.body)

    # Break any oversized paragraph down into sentences so no unit dwarfs the target.
    normalized: list[str] = []
    for unit in units:
        if estimate_tokens(unit) > target_tokens:
            normalized.extend(_split_sentences(unit))
        else:
            normalized.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in normalized:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_tokens:
            chunks.append("\n\n".join(current))
            # Start the next chunk with a sentence-level overlap tail for context.
            overlap = _take_overlap(current, overlap_tokens)
            current = overlap[:]
            current_tokens = sum(estimate_tokens(u) for u in current)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [
        Chunk(
            id=f"{article.id}::{i}",
            article_id=article.id,
            title=article.title,
            category=article.category,
            text=text,
            ordinal=i,
        )
        for i, text in enumerate(chunks)
    ]


def _take_overlap(units: list[str], overlap_tokens: int) -> list[str]:
    """Return a trailing slice of ``units`` worth roughly ``overlap_tokens``."""
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        tail.insert(0, unit)
        total += estimate_tokens(unit)
        if total >= overlap_tokens:
            break
    return tail
