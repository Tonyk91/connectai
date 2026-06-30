"""Unit tests for RRF fusion and tokenization (no models required)."""

from __future__ import annotations

from connectai.chunking import Chunk
from connectai.retrieval import rrf_fuse, tokenize


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, article_id=cid, title=cid, category="c", text=cid, ordinal=0)


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Top-Up FAILED, why?") == ["top", "up", "failed", "why"]


def test_rrf_rewards_agreement_between_lists() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    vector = [b, a, c]
    keyword = [b, c, a]  # b is rank 0 in both lists, so it must win
    fused = rrf_fuse([vector, keyword])
    assert fused[0].chunk.id == "b"
    # all unique ids preserved, no duplicates
    assert sorted(sc.chunk.id for sc in fused) == ["a", "b", "c"]


def test_rrf_dedupes_same_chunk_across_lists() -> None:
    a = _chunk("a")
    fused = rrf_fuse([[a], [a], [a]])
    assert len(fused) == 1
    # three contributions at rank 0 => 3 * 1/(60+1)
    assert abs(fused[0].score - 3 * (1 / 61)) < 1e-9


def test_rrf_orders_by_descending_score() -> None:
    a, b = _chunk("a"), _chunk("b")
    fused = rrf_fuse([[a, b]])
    assert [sc.chunk.id for sc in fused] == ["a", "b"]
    assert fused[0].score >= fused[1].score


def test_rrf_empty_input() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []
