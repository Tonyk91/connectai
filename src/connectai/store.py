"""ChromaDB-backed vector store.

Embeddings are computed by ConnectAI and passed in explicitly, so Chroma is used
purely as a persistent ANN index (cosine space). The store also exposes the full
chunk corpus, which the BM25 keyword retriever needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings

from .chunking import Chunk
from .config import Config


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with a retriever score (higher is better)."""

    chunk: Chunk
    score: float


def _chunk_from_record(doc: str, meta: Any, chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        article_id=str(meta.get("article_id", "")),
        title=str(meta.get("title", "")),
        category=str(meta.get("category", "")),
        text=doc,
        ordinal=int(meta.get("ordinal", 0)),
    )


class ConnectAIStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, config: Config) -> None:
        self._config = config
        config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(config.chroma_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

    def _collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop the collection so ingest starts from a clean slate."""
        try:
            self._client.delete_collection(self._config.collection_name)
        except Exception:
            pass

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        collection = self._collection()
        collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "article_id": c.article_id,
                    "title": c.title,
                    "category": c.category,
                    "ordinal": c.ordinal,
                }
                for c in chunks
            ],
        )

    def count(self) -> int:
        return self._collection().count()

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to build the BM25 index)."""
        result = self._collection().get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            _chunk_from_record(doc, meta, cid)
            for cid, doc, meta in zip(ids, docs, metas, strict=False)
        ]

    def query(self, embedding: list[float], top_n: int) -> list[ScoredChunk]:
        """Vector-search the collection; score = cosine similarity (1 - distance)."""
        result = self._collection().query(
            query_embeddings=[embedding],
            n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        scored: list[ScoredChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            chunk = _chunk_from_record(doc, meta, cid)
            scored.append(ScoredChunk(chunk=chunk, score=1.0 - float(dist)))
        return scored
