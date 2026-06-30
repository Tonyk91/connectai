"""Interactive command-line chat with ConnectAI.

Run with ``python -m connectai.cli``. Type a question and get a grounded answer
with its sources; type ``exit`` or Ctrl-D to quit.
"""

from __future__ import annotations

from .pipeline import Pipeline


def main() -> None:
    print("ConnectAI support chat — type 'exit' to quit.\n")
    pipeline = Pipeline()
    if pipeline.store.count() == 0:
        print("Index is empty. Run `python -m connectai.ingest` first.")
        return

    while True:
        try:
            query = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = pipeline.answer(query)
        print(f"\nbot > {result.answer}")
        if result.sources:
            cites = ", ".join(f"{s.article_id} ({s.score:.2f})" for s in result.sources)
            print(f"      sources: {cites}")
        print(
            f"      [{result.model} · {result.latency_ms:.0f} ms · "
            f"${result.est_cost_usd:.6f}]\n"
        )


if __name__ == "__main__":
    main()
