# Voice-Driven Quote Assistant

A **voice-first, retrieval-grounded** assistant that completes quotes and answers *who / source / disputed* strictly from a **Neo4j** graph (no hallucinations). Spoken input → Whisper ASR → fragment extraction → Neo4j full-text search (`quoteTextFT`) + rerank → **verbatim** answer via TTS.

## Features
- Neo4j graph: `Quote` ↔ `Person` via `SAID_BY` / `ABOUT` / `MISATTRIBUTED_TO` / `DISPUTED_WITH`
- Full-text index on `:Quote(text)` + lightweight reranker for robust fragment matching
- WebRTC VAD + Faster-Whisper (tiny/small); optional LLM for fuzzy intent (`USE_LLM_INTENTS=1`)
- Per-user TTS preferences and basic speaker ID

## Quickstart

### Step 1 — API (FastAPI + Neo4j)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements-step1.txt

$env:NEO4J_URI      = "neo4j://127.0.0.1:7687"
$env:NEO4J_USER     = "neo4j"
$env:NEO4J_PASSWORD = "<your-password>"
$env:NEO4J_DATABASE = "neo4j"   # or 'quotes_db'

uvicorn API:app --reload --port 8000
```
Test: `curl "http://127.0.0.1:8000/complete?fragment=two%20things%20are%20infinite"`

### Step 2 — Client (ASR + Router + TTS)
```bash
conda create -n step2 python=3.11 -y
conda activate step2
pip install -r Step_2/requirements.txt

conda env config vars set STEP1_API_BASE="http://127.0.0.1:8000" ASR_MODEL=small SIM_THRESHOLD=0.75
conda deactivate && conda activate step2

python Step_2/main.py
```

## Notes
- Keep secrets in env vars or a local `.env` (do **not** commit credentials).
- Avoid committing large models/audio (>100 MB). Use Git LFS or keep them local.

