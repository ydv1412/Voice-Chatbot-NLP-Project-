# Voice-Driven Quote Assistant (NLP Project)

A **voice-first, retrieval-grounded** assistant that completes quotes and answers *who / source / disputed* strictly from a **Neo4j** graph (no hallucinations).  
Spoken input → Whisper ASR → fragment extraction → Neo4j full‑text search (`quoteTextFT`) + rerank → **verbatim** answer via TTS.

---

## Features
- **Grounded retrieval:** Neo4j graph with `Quote` ↔ `Person` via `SAID_BY` / `ABOUT` / `MISATTRIBUTED_TO` / `DISPUTED_WITH`.
- **Fast search:** Lucene-backed full‑text index on `:Quote(text)` + lightweight reranker for robust partial fragments.
- **Voice pipeline:** WebRTC VAD + Faster-Whisper (tiny/small). Optional LLM router for fuzzy intents (`USE_LLM_INTENTS=1`).
- **Personalization:** Basic speaker ID and per-user TTS preferences (voice/rate/volume).

---

## Repository Layout (typical)
```
.
├─ Step_2/                      # Step 2: Client (ASR + Router + TTS)
│  ├─ main.py
|  ├─ asr.py
|  ├─ audio_utils.py
|  ├─ session.py.py
|  ├─ llm.py
|  ├─ user_prefs.py 
|  ├─ speaker_id.py
|  ├─ config.py
|  ├─ retriever.py
|  ├─ intent_mapper.py
|  ├─ dialogue.py
│  └─ requirements.txt
├─ notebooks                    #  data extraction / preprocessing notebooks
├─ .gitignore
└─ README.md

### To run the project
1) **Create env & install dependencies**
```bash
conda create -n step2 python=3.11 -y
conda activate step2
pip install -r Step_2/requirements.txt
```
2) **Set variables (persist in conda)**
```bash
conda env config vars set \
  STEP1_API_BASE="http://127.0.0.1:8000" \
  ASR_MODEL=small \
  SIM_THRESHOLD=0.75
# Optional (slower): enable LLM intent mapper
conda env config vars set USE_LLM_INTENTS=1
conda deactivate && conda activate step2
```
3) **Run**
```bash
python Step_2/main.py
```

Note:
You should have a graph database API to fetch the records from.


