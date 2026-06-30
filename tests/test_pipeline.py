"""Pipeline gate + grounded-generation tests. Claude is mocked; no real API calls.

These tests avoid loading the embedding/reranker models — that heavy, end-to-end
path is exercised by the eval harness in CI instead.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

from connectai.chunking import Chunk
from connectai.config import load_config
from connectai.generate import REFUSAL_MESSAGE, generate_answer
from connectai.pipeline import Pipeline
from connectai.store import ScoredChunk


def _scored(article_id: str, text: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        id=f"{article_id}::0",
        article_id=article_id,
        title=article_id,
        category="c",
        text=text,
        ordinal=0,
    )
    return ScoredChunk(chunk=chunk, score=score)


def _config(tmp_path: Path, **overrides: object):
    overrides.setdefault("anthropic_api_key", "")
    base = load_config()
    return dataclasses.replace(
        base,
        chroma_dir=tmp_path / "chroma",
        log_file=tmp_path / "log.jsonl",
        **overrides,
    )


# --- Gate --------------------------------------------------------------------


def test_gate_filters_below_threshold(tmp_path: Path) -> None:
    config = _config(tmp_path, min_score=0.5)
    pipeline = Pipeline(config)
    candidates = [
        _scored("a", "relevant", 0.9),
        _scored("b", "weak", 0.4),
        _scored("c", "irrelevant", 0.1),
    ]
    gated = pipeline._gate(candidates)
    assert [sc.chunk.article_id for sc in gated] == ["a"]


def test_gate_keeps_all_when_above_threshold(tmp_path: Path) -> None:
    config = _config(tmp_path, min_score=0.2)
    pipeline = Pipeline(config)
    candidates = [_scored("a", "x", 0.9), _scored("b", "y", 0.3)]
    assert len(pipeline._gate(candidates)) == 2


# --- Grounded generation (fallback, no key) ----------------------------------


def test_fallback_answer_is_grounded_and_cited(tmp_path: Path) -> None:
    config = _config(tmp_path)
    chunks = [_scored("refund-policy", "Unused call credit can be refunded within 12 months.", 0.8)]
    result = generate_answer("can I refund unused credit?", chunks, config)
    assert result.grounded is True
    assert result.refused is False
    assert result.citations == ["refund-policy"]
    assert result.model == "fallback-extractive"
    assert result.answer


def test_empty_context_refuses(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = generate_answer("what is the capital of France?", [], config)
    assert result.refused is True
    assert result.grounded is False
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


# --- Real-Claude path with a mocked SDK --------------------------------------


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch):
    created = {}

    class _Block:
        type = "text"
        text = "You can refund unused credit per the refund policy."

    class _Usage:
        input_tokens = 123
        output_tokens = 45

    class _Response:
        content = [_Block()]
        usage = _Usage()

    class _Messages:
        def create(self, **kwargs: object) -> _Response:
            created.update(kwargs)
            return _Response()

    class _Anthropic:
        def __init__(self, api_key: str = "") -> None:
            self.messages = _Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = _Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return created


def test_real_claude_path_uses_model_and_usage(tmp_path: Path, fake_anthropic: dict) -> None:
    config = _config(tmp_path, anthropic_api_key="test-key", claude_model="claude-sonnet-4-6")
    chunks = [_scored("refund-policy", "Unused call credit can be refunded.", 0.8)]
    result = generate_answer("refund unused credit?", chunks, config)
    assert result.model == "claude-sonnet-4-6"
    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert result.grounded is True
    assert fake_anthropic["model"] == "claude-sonnet-4-6"
