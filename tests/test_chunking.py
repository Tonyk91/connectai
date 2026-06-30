"""Unit tests for article parsing and token-aware chunking."""

from __future__ import annotations

from connectai.chunking import (
    article_from_markdown,
    chunk_article,
    estimate_tokens,
    parse_frontmatter,
)

RAW = """---
id: demo-article
title: Demo Article
category: billing
---

# Demo Article

First paragraph with a few words about billing and payments.

Second paragraph that talks about refunds and invoices in some detail.
"""


def test_estimate_tokens_scales_with_words() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three four") > 3


def test_parse_frontmatter_extracts_fields() -> None:
    meta, body = parse_frontmatter(RAW)
    assert meta["id"] == "demo-article"
    assert meta["title"] == "Demo Article"
    assert meta["category"] == "billing"
    assert body.startswith("# Demo Article")
    assert "---" not in body.splitlines()[0]


def test_parse_frontmatter_without_block() -> None:
    meta, body = parse_frontmatter("no frontmatter here")
    assert meta == {}
    assert body == "no frontmatter here"


def test_article_from_markdown_uses_fallback_id() -> None:
    article = article_from_markdown("just body text", fallback_id="fallback")
    assert article.id == "fallback"
    assert article.category == "general"


def test_chunk_article_produces_chunks_with_stable_ids() -> None:
    article = article_from_markdown(RAW, fallback_id="x")
    chunks = chunk_article(article, target_tokens=30, overlap_tokens=8)
    assert len(chunks) >= 2
    assert all(c.article_id == "demo-article" for c in chunks)
    assert [c.id for c in chunks] == [f"demo-article::{i}" for i in range(len(chunks))]
    assert all(c.ordinal == i for i, c in enumerate(chunks))


def test_chunk_article_small_input_single_chunk() -> None:
    article = article_from_markdown("---\nid: a\ntitle: A\ncategory: c\n---\nShort body.", "a")
    chunks = chunk_article(article, target_tokens=400, overlap_tokens=60)
    assert len(chunks) == 1
    assert "Short body." in chunks[0].text
