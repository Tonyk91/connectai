"""Grounded answer generation.

Answers are generated strictly from retrieved context and always carry a source
citation. With an ANTHROPIC_API_KEY, Claude writes the answer; without one, a
deterministic extractive fallback runs so eval/tests/CI need no secrets and incur
no cost. If no context survives the gate, the agent refuses rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .store import ScoredChunk

REFUSAL_MESSAGE = "I don't have information about that. Please contact our support team."

_SYSTEM_PROMPT = (
    "You are ConnectAI, a customer-support assistant for an international telecom "
    "service (international calling, mobile top-ups, billing). Answer ONLY using the "
    "provided context passages. If the context does not contain the answer, reply "
    f'exactly: "{REFUSAL_MESSAGE}" Be concise, friendly and practical. Do not invent '
    "policies, prices, or steps that are not in the context."
)


@dataclass(frozen=True)
class GenerationResult:
    """The model's answer plus grounding metadata."""

    answer: str
    citations: list[str] = field(default_factory=list)
    model: str = "fallback-extractive"
    grounded: bool = True
    refused: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for sc in chunks:
        c = sc.chunk
        blocks.append(f"[source: {c.article_id} — {c.title}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _citations(chunks: list[ScoredChunk]) -> list[str]:
    seen: list[str] = []
    for sc in chunks:
        aid = sc.chunk.article_id
        if aid not in seen:
            seen.append(aid)
    return seen


def _extractive_answer(query: str, chunks: list[ScoredChunk]) -> str:
    """Deterministic, key-free answer: the most on-topic sentences from the top chunk."""
    import re

    top = chunks[0].chunk
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    # Drop markdown heading lines and collapse whitespace so the extractive
    # answer reads as clean prose rather than raw markdown.
    body_lines = [ln for ln in top.text.splitlines() if not ln.lstrip().startswith("#")]
    clean = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    scored = sorted(
        sentences,
        key=lambda s: len(query_terms & set(re.findall(r"[a-z0-9]+", s.lower()))),
        reverse=True,
    )
    picked = scored[:3] if scored else sentences[:2]
    # Preserve original order for readability.
    ordered = [s for s in sentences if s in picked]
    return " ".join(ordered) if ordered else top.text[:400]


def generate_answer(query: str, chunks: list[ScoredChunk], config: Config) -> GenerationResult:
    """Generate a grounded answer from gated ``chunks`` (already above threshold)."""
    if not chunks:
        return GenerationResult(answer=REFUSAL_MESSAGE, grounded=False, refused=True)

    citations = _citations(chunks)

    if not config.use_real_claude:
        answer = _extractive_answer(query, chunks)
        return GenerationResult(
            answer=answer,
            citations=citations,
            model="fallback-extractive",
            grounded=True,
            input_tokens=len(query.split()) + sum(len(sc.chunk.text.split()) for sc in chunks),
            output_tokens=len(answer.split()),
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=config.anthropic_api_key)
    context = _format_context(chunks)
    user_prompt = (
        f"Context passages:\n\n{context}\n\n"
        f"Customer question: {query}\n\n"
        "Answer using only the context above and mention which source you used."
    )
    response = client.messages.create(
        model=config.claude_model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer = "".join(block.text for block in response.content if block.type == "text").strip()
    refused = answer.strip() == REFUSAL_MESSAGE
    return GenerationResult(
        answer=answer,
        citations=[] if refused else citations,
        model=config.claude_model,
        grounded=not refused,
        refused=refused,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
