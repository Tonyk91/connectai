"""Hybrid retrieval: dense vector search fused with BM25 keyword search.

The two ranked lists are combined with Reciprocal Rank Fusion (RRF), which is
robust to the two retrievers using different score scales. The fused top-N is
then handed to the reranker.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .config import Config
from .embeddings import Embedder
from .store import ConnectAIStore, ScoredChunk

_RRF_K = 60
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenizer shared by BM25 indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(ranked_lists: list[list[Chunk]], k: int = _RRF_K) -> list[ScoredChunk]:
    """Fuse several ranked lists of chunks into one, scored by RRF.

    Each list contributes ``1 / (k + rank)`` per chunk (rank is 0-based). Chunks
    are de-duplicated by id and returned sorted by descending fused score.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            seen.setdefault(chunk.id, chunk)
    fused = [ScoredChunk(chunk=seen[cid], score=score) for cid, score in scores.items()]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused


class HybridRetriever:
    """Combines Chroma vector search with an in-memory BM25 index."""

    def __init__(self, config: Config, store: ConnectAIStore, embedder: Embedder) -> None:
        self._config = config
        self._store = store
        self._embedder = embedder
        self._corpus: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            self._corpus = self._store.all_chunks()
            tokenized = [tokenize(c.text) for c in self._corpus] or [[""]]
            self._bm25 = BM25Okapi(tokenized)

    def _vector_candidates(self, query: str, top_n: int) -> list[Chunk]:
        embedding = self._embedder.embed([query])[0]
        return [sc.chunk for sc in self._store.query(embedding, top_n)]

    def _bm25_candidates(self, query: str, top_n: int) -> list[Chunk]:
        self._ensure_bm25()
        if not self._corpus:
            return []
        assert self._bm25 is not None
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self._corpus)), key=lambda i: scores[i], reverse=True)
        return [self._corpus[i] for i in ranked[:top_n]]

    def retrieve(self, query: str, top_n: int | None = None) -> list[ScoredChunk]:
        """Return the RRF-fused top-N candidates for ``query``."""
        n = top_n or self._config.retrieval_top_n
        vector = self._vector_candidates(query, n)
        keyword = self._bm25_candidates(query, n)
        return rrf_fuse([vector, keyword])[:n]
