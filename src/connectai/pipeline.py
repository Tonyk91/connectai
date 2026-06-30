"""End-to-end RAG pipeline orchestration.

Wires the stages together: hybrid retrieval -> cross-encoder rerank -> relevance
gate -> grounded generation, emitting one observability record per call. Heavy
components (models, store) are built once and reused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Config, load_config
from .embeddings import build_embedder
from .generate import GenerationResult, generate_answer
from .observability import RequestLog, estimate_cost, log_request, now_iso
from .rerank import CrossEncoderReranker
from .retrieval import HybridRetriever
from .store import ConnectAIStore, ScoredChunk


@dataclass(frozen=True)
class Source:
    """A cited source surfaced to the caller."""

    article_id: str
    title: str
    score: float


@dataclass(frozen=True)
class ChatResponse:
    """The full result of a /chat call."""

    answer: str
    citations: list[str]
    sources: list[Source]
    model: str
    grounded: bool
    refused: bool
    latency_ms: float
    est_cost_usd: float
    retrieved: int = 0
    reranked: int = 0
    debug: dict[str, object] = field(default_factory=dict)


class Pipeline:
    """Reusable RAG pipeline; construct once, call :meth:`answer` per query."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.store = ConnectAIStore(self.config)
        self.embedder = build_embedder(self.config)
        self.retriever = HybridRetriever(self.config, self.store, self.embedder)
        self.reranker = CrossEncoderReranker(self.config)

    def retrieve_and_rerank(self, query: str) -> list[ScoredChunk]:
        """Retrieval + rerank without generation (used by the eval harness)."""
        candidates = self.retriever.retrieve(query)
        return self.reranker.rerank(query, candidates)

    def _gate(self, reranked: list[ScoredChunk]) -> list[ScoredChunk]:
        return [sc for sc in reranked if sc.score >= self.config.min_score]

    def answer(self, query: str) -> ChatResponse:
        """Run the full pipeline for ``query`` and log the request."""
        start = time.perf_counter()
        candidates = self.retriever.retrieve(query)
        reranked = self.reranker.rerank(query, candidates)
        gated = self._gate(reranked)

        result: GenerationResult = generate_answer(query, gated, self.config)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        cost = estimate_cost(result.model, result.input_tokens, result.output_tokens)

        sources = [
            Source(article_id=sc.chunk.article_id, title=sc.chunk.title, score=round(sc.score, 4))
            for sc in gated
        ]

        log_request(
            RequestLog(
                timestamp=now_iso(),
                query=query,
                latency_ms=latency_ms,
                retrieved_chunks=len(candidates),
                reranked_k=len(gated),
                model=result.model,
                est_cost_usd=cost,
                gated=result.refused,
                citations=result.citations,
            ),
            self.config.log_file,
        )

        return ChatResponse(
            answer=result.answer,
            citations=result.citations,
            sources=sources,
            model=result.model,
            grounded=result.grounded,
            refused=result.refused,
            latency_ms=latency_ms,
            est_cost_usd=cost,
            retrieved=len(candidates),
            reranked=len(gated),
        )
