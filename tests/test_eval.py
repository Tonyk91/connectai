"""Unit tests for the eval metric math."""

from __future__ import annotations

from connectai.eval import hit_at_k, recall_at_k, reciprocal_rank


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_reciprocal_rank_missing() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_hit_at_k_within_and_outside_window() -> None:
    assert hit_at_k(["x", "a", "y"], {"a"}, k=5) == 1.0
    assert hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_at_k_partial_and_full() -> None:
    assert recall_at_k(["a", "b", "z"], {"a", "b"}, k=5) == 1.0
    assert recall_at_k(["a", "z", "y"], {"a", "b"}, k=5) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, k=1) == 0.5


def test_recall_at_k_no_expected() -> None:
    assert recall_at_k(["a"], set(), k=5) == 0.0
