"""Retrieval evaluation harness with a CI regression gate.

Runs a labelled test set through retrieval + rerank and reports Hit Rate@K, MRR
and Recall@K, plus refusal accuracy on deliberately out-of-corpus queries. Writes
``eval_results.json`` and ``eval_report.md``. Exits non-zero if Hit Rate drops
below the configured threshold, which fails CI.

Run with ``python -m connectai.eval``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, load_config
from .ingest import ingest
from .pipeline import Pipeline

DEFAULT_TESTSET = Path("eval/testset.json")


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_articles: list[str]
    type: str  # "in_corpus" or "out_of_corpus"


# --- Pure metric helpers (unit-tested) ---------------------------------------


def reciprocal_rank(ranked_ids: list[str], expected: set[str]) -> float:
    """1 / rank of the first relevant id (1-based), or 0.0 if none are present."""
    for idx, aid in enumerate(ranked_ids):
        if aid in expected:
            return 1.0 / (idx + 1)
    return 0.0


def hit_at_k(ranked_ids: list[str], expected: set[str], k: int) -> float:
    """1.0 if any expected id is within the top-k, else 0.0."""
    return 1.0 if expected & set(ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: list[str], expected: set[str], k: int) -> float:
    """Fraction of expected ids found within the top-k."""
    if not expected:
        return 0.0
    return len(expected & set(ranked_ids[:k])) / len(expected)


# --- Harness -----------------------------------------------------------------


def load_testset(path: Path) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            query=row["query"],
            expected_articles=row.get("expected_articles", []),
            type=row.get("type", "in_corpus"),
        )
        for row in data
    ]


def evaluate(config: Config | None = None, testset_path: Path = DEFAULT_TESTSET) -> dict[str, object]:
    """Run the full evaluation and return a results dict."""
    config = config or load_config()
    pipeline = Pipeline(config)

    # Self-sufficient: build the index if it has not been ingested yet.
    if pipeline.store.count() == 0:
        ingest(config)
        pipeline = Pipeline(config)

    k = config.rerank_top_k
    cases = load_testset(testset_path)

    per_query: list[dict[str, object]] = []
    hits: list[float] = []
    rrs: list[float] = []
    recalls: list[float] = []
    refusal_correct = 0
    refusal_total = 0

    for case in cases:
        reranked = pipeline.retrieve_and_rerank(case.query)
        ranked_ids = [sc.chunk.article_id for sc in reranked]
        top_score = reranked[0].score if reranked else 0.0
        expected = set(case.expected_articles)

        if case.type == "out_of_corpus":
            refusal_total += 1
            refused = top_score < config.min_score
            if refused:
                refusal_correct += 1
            per_query.append(
                {
                    "query": case.query,
                    "type": case.type,
                    "top_score": round(top_score, 4),
                    "refused": refused,
                    "correct": refused,
                }
            )
            continue

        rr = reciprocal_rank(ranked_ids, expected)
        hit = hit_at_k(ranked_ids, expected, k)
        rec = recall_at_k(ranked_ids, expected, k)
        hits.append(hit)
        rrs.append(rr)
        recalls.append(rec)
        per_query.append(
            {
                "query": case.query,
                "type": case.type,
                "expected": case.expected_articles,
                "top_ranked": ranked_ids[:k],
                "hit": hit,
                "reciprocal_rank": round(rr, 4),
                "recall": round(rec, 4),
            }
        )

    n = len(hits)
    summary = {
        "k": k,
        "in_corpus_queries": n,
        "out_of_corpus_queries": refusal_total,
        "hit_rate": round(sum(hits) / n, 4) if n else 0.0,
        "mrr": round(sum(rrs) / n, 4) if n else 0.0,
        "recall_at_k": round(sum(recalls) / n, 4) if n else 0.0,
        "refusal_accuracy": round(refusal_correct / refusal_total, 4) if refusal_total else 0.0,
        "hit_rate_threshold": config.eval_hit_rate_threshold,
    }
    summary["passed"] = bool(summary["hit_rate"] >= config.eval_hit_rate_threshold)

    return {"summary": summary, "per_query": per_query}


def write_report(results: dict[str, object], json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    s = results["summary"]
    assert isinstance(s, dict)
    status = "✅ PASS" if s["passed"] else "❌ FAIL"
    md = f"""# ConnectAI — Evaluation Report

**Status:** {status} (gate: Hit Rate ≥ {s['hit_rate_threshold']})

| Metric | Value |
|---|---|
| Hit Rate@{s['k']} | {s['hit_rate']} |
| MRR | {s['mrr']} |
| Recall@{s['k']} | {s['recall_at_k']} |
| Refusal accuracy (out-of-corpus) | {s['refusal_accuracy']} |
| In-corpus queries | {s['in_corpus_queries']} |
| Out-of-corpus queries | {s['out_of_corpus_queries']} |

Generated by `python -m connectai.eval`.
"""
    md_path.write_text(md, encoding="utf-8")


def main() -> None:
    config = load_config()
    results = evaluate(config)
    write_report(results, Path("eval_results.json"), Path("eval_report.md"))

    summary = results["summary"]
    assert isinstance(summary, dict)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        print(
            f"\nEval gate FAILED: Hit Rate {summary['hit_rate']} "
            f"< threshold {summary['hit_rate_threshold']}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("\nEval gate passed.")


if __name__ == "__main__":
    main()
