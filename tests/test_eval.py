"""Unit tests for the eval metric math."""

from __future__ import annotations

from connectai.eval import gate_failures, hit_at_k, recall_at_k, reciprocal_rank


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


def _summary(hit_rate: float, refusal: float) -> dict[str, object]:
    return {
        "hit_rate": hit_rate,
        "hit_rate_threshold": 0.70,
        "refusal_accuracy": refusal,
        "refusal_threshold": 1.00,
    }


def test_gate_passes_when_both_thresholds_met() -> None:
    assert gate_failures(_summary(0.95, 1.0)) == []


def test_gate_fails_on_refusal_alone() -> None:
    failures = gate_failures(_summary(1.0, 0.5))
    assert len(failures) == 1
    assert failures[0].startswith("Refusal accuracy")


def test_gate_fails_on_hit_rate_alone() -> None:
    failures = gate_failures(_summary(0.5, 1.0))
    assert len(failures) == 1
    assert failures[0].startswith("Hit Rate")
