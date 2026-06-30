"""Central configuration, loaded once from the environment.

All tunables live here so the rest of the codebase stays free of ``os.getenv`` calls.
Values fall back to sensible defaults that let eval/tests/CI run with zero secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _get(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    # Generation
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: _get("CLAUDE_MODEL", "claude-sonnet-4-6"))

    # Embeddings
    embedding_backend: str = field(
        default_factory=lambda: _get("EMBEDDING_BACKEND", "sentence-transformers")
    )
    embedding_model: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    openai_embedding_model: str = field(
        default_factory=lambda: _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Reranker
    rerank_model: str = field(default_factory=lambda: _get("RERANK_MODEL", "BAAI/bge-reranker-base"))

    # Retrieval / gate
    retrieval_top_n: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_N", 10))
    rerank_top_k: int = field(default_factory=lambda: _get_int("RERANK_TOP_K", 5))
    min_score: float = field(default_factory=lambda: _get_float("MIN_SCORE", 0.51))

    # Storage
    chroma_dir: Path = field(default_factory=lambda: Path(_get("CHROMA_DIR", "data/chroma")))
    kb_dir: Path = field(default_factory=lambda: Path(_get("KB_DIR", "data/kb")))
    log_file: Path = field(default_factory=lambda: Path(_get("LOG_FILE", "logs/requests.jsonl")))

    # Eval gate
    eval_hit_rate_threshold: float = field(
        default_factory=lambda: _get_float("EVAL_HIT_RATE_THRESHOLD", 0.70)
    )

    # Chunking
    chunk_target_tokens: int = field(default_factory=lambda: _get_int("CHUNK_TARGET_TOKENS", 180))
    chunk_overlap_tokens: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP_TOKENS", 40))

    @property
    def use_real_claude(self) -> bool:
        """True when a key is present, so generation calls the real Claude API."""
        return bool(self.anthropic_api_key)

    @property
    def collection_name(self) -> str:
        return "connectai_kb"


def load_config() -> Config:
    """Build a :class:`Config` from the current environment."""
    return Config()
