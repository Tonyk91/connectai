"""Build the vector index from the knowledge base.

Reads every markdown article under ``KB_DIR``, chunks it, embeds the chunks and
writes them to the Chroma collection. Run with ``python -m connectai.ingest``.
"""

from __future__ import annotations

import sys

from .chunking import Article, Chunk, article_from_markdown, chunk_article
from .config import Config, load_config
from .embeddings import build_embedder
from .store import ConnectAIStore


def load_articles(config: Config) -> list[Article]:
    """Load all markdown articles from the knowledge-base directory."""
    if not config.kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {config.kb_dir}")
    articles: list[Article] = []
    for path in sorted(config.kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        articles.append(article_from_markdown(raw, fallback_id=path.stem))
    return articles


def build_chunks(articles: list[Article], config: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(
            chunk_article(
                article,
                target_tokens=config.chunk_target_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
            )
        )
    return chunks


def ingest(config: Config | None = None) -> int:
    """Re-ingest the whole knowledge base; returns the number of chunks stored."""
    config = config or load_config()
    articles = load_articles(config)
    chunks = build_chunks(articles, config)

    embedder = build_embedder(config)
    embeddings = embedder.embed([c.text for c in chunks])

    store = ConnectAIStore(config)
    store.reset()
    store.add_chunks(chunks, embeddings)
    return len(chunks)


def main() -> None:
    config = load_config()
    print(f"Ingesting knowledge base from {config.kb_dir} ...")
    count = ingest(config)
    print(f"Done: {count} chunks stored in Chroma at {config.chroma_dir}")
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
