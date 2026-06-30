"""FastAPI surface for ConnectAI.

Endpoints:
- ``POST /chat``   — answer a customer question (grounded, with citations)
- ``GET  /metrics`` — aggregated observability metrics from the request log
- ``GET  /health``  — readiness probe reporting indexed chunk count

The pipeline (and its models) is built once at startup and reused across requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import load_config
from .observability import load_metrics
from .pipeline import Pipeline

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ConnectAI", version="0.1.0", description="RAG customer-support agent")

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The customer's question")


class SourceModel(BaseModel):
    article_id: str
    title: str
    score: float


class ChatResponseModel(BaseModel):
    answer: str
    citations: list[str]
    sources: list[SourceModel]
    model: str
    grounded: bool
    refused: bool
    latency_ms: float
    est_cost_usd: float


@app.on_event("startup")
def _startup() -> None:
    # Warm the pipeline so the first request is not penalised with model loading.
    get_pipeline()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page chat UI."""
    return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "indexed_chunks": pipeline.store.count(),
        "generation": "claude" if pipeline.config.use_real_claude else "fallback",
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    config = load_config()
    return load_metrics(config.log_file)


@app.post("/chat", response_model=ChatResponseModel)
def chat(request: ChatRequest) -> ChatResponseModel:
    pipeline = get_pipeline()
    result = pipeline.answer(request.message)
    return ChatResponseModel(
        answer=result.answer,
        citations=result.citations,
        sources=[
            SourceModel(article_id=s.article_id, title=s.title, score=s.score)
            for s in result.sources
        ],
        model=result.model,
        grounded=result.grounded,
        refused=result.refused,
        latency_ms=result.latency_ms,
        est_cost_usd=result.est_cost_usd,
    )
