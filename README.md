# Voice-Driven Quote Assistant

This project builds a **voice-first, retrieval-grounded** assistant that completes quotes and answers *who / source / disputed* strictly from a **Neo4j** graph (no hallucinations).  
It has two parts: **Step 1** (FastAPI + Neo4j backend) and **Step 2** (client with ASR + routing + TTS).

---

## Pointers:-
1. **Data → Graph:** ~30k Wikiquote rows → cleaned to **27,799** quotes; loaded into Neo4j as `Quote` ↔ `Person` with `SAID_BY`, `ABOUT`, `MISATTRIBUTED_TO`, `DISPUTED_WITH`.
2. **Retrieval:** Neo4j full-text index `quoteTextFT` + simple reranker (token coverage + FT score + phrase bonus).
3. **Voice pipeline:** WebRTC **VAD** → **Whisper** ASR (tiny for quick commands, small for accuracy).
4. **Deterministic answers:** “Finish” returns the **verbatim** quote; metadata comes from stored relations (no LLM hallucinations).
5. **LLM usage (optional):** only for **fragment extraction** and **fuzzy intent** (`USE_LLM_INTENTS=1`).
6. **Personalization:** Speaker ID + per-user TTS (voice/rate/volume) via `pyttsx3` (Edge-TTS available).

---

## Software & Tools:-
1. [GitHub](https://github.com/)
2. [VS Code](https://code.visualstudio.com/)
3. [Git CLI](https://git-scm.com/)
4. [Neo4j 5.x](https://neo4j.com/)
5. [FastAPI + Uvicorn](https://fastapi.tiangolo.com/)
6. [Faster-Whisper & webrtcvad](https://github.com/SYSTRAN/faster-whisper)

---

## Create virtual environments

### Step 1 (API) – venv
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements-step1.txt


conda create -n step2 python=3.11 -y
conda activate step2
pip install -r Step_2/requirements.txt


conda env config vars set \
  STEP1_API_BASE="http://127.0.0.1:8000" \
  LLM_GGUF="C:\path\to\mistral-7b-instruct-v0.1.Q4_K_M.gguf" \
  ASR_MODEL=small \
  SIM_THRESHOLD=0.75
conda deactivate && conda activate step2
# (Optional) enable LLM intent mapper
conda env config vars set USE_LLM_INTENTS=1
conda deactivate && conda activate step2


conda activate step2
cd Step_2
python main.py
```
Voice-Driven-Quote-Assistant/
│── Step_2/
│   │── main.py
│   │── retriever.py
│   │── dialogue.py
│   │── llm.py
│   │── intent_mapper.py
│   │── asr.py
│   │── tts.py
│   │── audio_utils.py
│   │── session.py
│   │── speaker_id.py
│   │── user_prefs.py
│   │── config.py
│   │── requirements.txt
│── Extracted_File_Preprocessing.ipynb
│── Wikiquote_quote_extraction.ipynb
│── generate_test_dataset.py
│── .gitignore
│── README.md
