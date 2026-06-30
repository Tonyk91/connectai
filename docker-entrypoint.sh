#!/usr/bin/env bash
set -euo pipefail

# Build the index on first boot if the Chroma collection is empty, then serve.
python - <<'PY'
from connectai.config import load_config
from connectai.store import ConnectAIStore
from connectai.ingest import ingest

config = load_config()
if ConnectAIStore(config).count() == 0:
    print("No index found — running ingest ...", flush=True)
    n = ingest(config)
    print(f"Ingested {n} chunks.", flush=True)
else:
    print("Existing index found — skipping ingest.", flush=True)
PY

exec uvicorn connectai.api:app --host 0.0.0.0 --port 8000
