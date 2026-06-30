"""Embedding backends.

Default is a local ``sentence-transformers`` model so no API key is needed for
ingest, retrieval, eval or CI. An OpenAI backend is available via configuration.
Models are lazy-loaded the first time embeddings are requested.
"""

from __future__ import annotations

from typing import Any, Protocol

from .config import Config


class Embedder(Protocol):
    """Anything that can turn texts into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def name(self) -> str: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (no network at inference)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    @property
    def name(self) -> str:
        return self._model_name


class OpenAIEmbedder:
    """Embeddings via the OpenAI API (opt-in, requires OPENAI_API_KEY)."""

    def __init__(self, model_name: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai embedding backend")
        self._model_name = model_name
        self._api_key = api_key
        self._client: Any = None  # lazy

    def _ensure(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        assert self._client is not None
        response = self._client.embeddings.create(model=self._model_name, input=texts)
        return [item.embedding for item in response.data]

    @property
    def name(self) -> str:
        return self._model_name


def build_embedder(config: Config) -> Embedder:
    """Construct the configured embedding backend."""
    if config.embedding_backend == "openai":
        return OpenAIEmbedder(config.openai_embedding_model, config.openai_api_key)
    return SentenceTransformerEmbedder(config.embedding_model)
