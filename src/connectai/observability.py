"""Structured, per-request observability.

Every pipeline call emits one JSON line (to stdout and a JSONL file) capturing
latency, retrieval depth, model, estimated cost and whether the gate fired. The
/metrics endpoint aggregates that log so the system is observable in production.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# USD per 1M tokens. Fallback (key-free) generation is free.
_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "fallback-extractive": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate generation cost in USD for a known model (0.0 if unknown/free)."""
    in_price, out_price = _PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price, 6)


@dataclass(frozen=True)
class RequestLog:
    """One structured log line for a single /chat request."""

    timestamp: str
    query: str
    latency_ms: float
    retrieved_chunks: int
    reranked_k: int
    model: str
    est_cost_usd: float
    gated: bool
    citations: list[str]


def log_request(record: RequestLog, log_file: Path) -> None:
    """Append ``record`` as a JSON line to ``log_file`` and echo to stdout."""
    line = json.dumps(asdict(record), ensure_ascii=False)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, file=sys.stdout, flush=True)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[idx], 2)


def load_metrics(log_file: Path) -> dict[str, object]:
    """Aggregate the request log into summary metrics for /metrics."""
    if not log_file.exists():
        return {
            "requests": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "total_est_cost_usd": 0.0,
            "gated_rate": 0.0,
        }

    latencies: list[float] = []
    total_cost = 0.0
    gated = 0
    count = 0
    with log_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            latencies.append(float(row.get("latency_ms", 0.0)))
            total_cost += float(row.get("est_cost_usd", 0.0))
            if row.get("gated"):
                gated += 1

    avg_latency = round(sum(latencies) / count, 2) if count else 0.0
    return {
        "requests": count,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": _percentile(latencies, 95),
        "total_est_cost_usd": round(total_cost, 6),
        "gated_rate": round(gated / count, 3) if count else 0.0,
    }
