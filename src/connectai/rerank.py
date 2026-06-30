"""Cross-encoder reranking of fused candidates.

A bge-reranker cross-encoder scores each (query, chunk) pair jointly, which is
far more precise than the first-stage retrievers. Raw logits are squashed with a
sigmoid into [0, 1] so the gate threshold (MIN_SCORE) is interpretable.
"""

from __future__ import annotations

import math
from typing import Any

from .config import Config
from .store import ScoredChunk


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class CrossEncoderReranker:
    """Reranks candidates with a sentence-transformers CrossEncoder."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None  # lazy

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._config.rerank_model)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int | None = None) -> list[ScoredChunk]:
        """Return ``candidates`` rescored by the cross-encoder, best first."""
        if not candidates:
            return []
        k = top_k or self._config.rerank_top_k
        self._ensure()
        assert self._model is not None
        pairs = [(query, sc.chunk.text) for sc in candidates]
        raw = self._model.predict(pairs)
        rescored = [
            ScoredChunk(chunk=sc.chunk, score=_sigmoid(float(score)))
            for sc, score in zip(candidates, raw, strict=False)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:k]
