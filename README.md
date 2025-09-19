# Voice-Driven Quote Assistant (NLP Chatbot Project)

A **voice-first, retrieval-grounded** assistant that completes quotes and answers *who / source / disputed* strictly from a **Neo4j** graph (no hallucinations).  
**Step 1** exposes a Neo4j-backed API. **Step 2** is the client (ASR + routing + TTS) that calls the API.

---

## 🗂️ Repository Structure

```
.
├─ API.py                       # Step 1: FastAPI app (Neo4j autocomplete/complete)
├─ requirements-step1.txt       # Step 1 dependencies
├─ Step_2/                      # Step 2: Client (ASR + Router + TTS)
│  ├─ main.py
│  ├─ retriever.py
│  ├─ dialogue.py
│  ├─ llm.py
│  ├─ intent_mapper.py
│  ├─ asr.py
│  ├─ tts.py
│  ├─ audio_utils.py
│  ├─ session.py
│  ├─ speaker_id.py
│  ├─ user_prefs.py
│  ├─ config.py
│  ├─ eval.py
│  ├─ eval_text.py
│  └─ requirements.txt          # Step 2 dependencies
├─ Extracted_File_Preprocessing.ipynb
├─ Wikiquote_quote_extraction.ipynb
├─ generate_test_dataset.py
├─ test_dataset/                # (optional) small sample data for tests
├─ Extracted dataset versions/  # (optional) local data exports (ignored by git)
├─ .gitignore
└─ README.md
```

> ⚠️ **Do not** commit large models/audio or real credentials. Use `.gitignore` (already included) and consider Git LFS for files >100 MB.

---

## 📊 Data & Graph (from the report)

- Wikiquote dump → ~30,000 raw rows.  
- After preprocessing (cleanup, dedupe, length ≤ 1000 chars): **27,799** quotes with:
  - `text`, `source`, `author → (status, target_name)`, `heading_context`.
- **Graph schema (Neo4j 5.26.9):**
  - **Quote**: `id`, `text`, `source`, `heading_context`, `status`, `target_name`
  - **Person**: `id`, `name`
  - Relations: `SAID_BY`, `ABOUT`, `MISATTRIBUTED_TO`, `DISPUTED_WITH`
- Full-text index on `Quote.text`: **`quoteTextFT`**.

---

## 🚀 Quickstart

### 0) Requirements
- Python **3.11**
- Neo4j **5.x** (you used 5.26.9)
- (Optional) Conda for Step 2

---

## 1) Step 1 — Neo4j API

### 1.1 Create venv & install deps (Windows PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# create the file, or edit if it already exists
ni requirements-step1.txt -ItemType File
notepad requirements-step1.txt
```

Put this in **requirements-step1.txt**:
```
fastapi==0.115.0
uvicorn[standard]==0.30.5
neo4j==5.23.0
python-dotenv==1.0.1
```

Install:
```powershell
pip install -r requirements-step1.txt
```

### 1.2 Configure env vars
```powershell
$env:NEO4J_URI      = "neo4j://127.0.0.1:7687"
$env:NEO4J_USER     = "neo4j"
$env:NEO4J_PASSWORD = "<your-password-here>"   # DO NOT commit this
$env:NEO4J_DATABASE = "neo4j"                  # or 'quotes_db' if that's your DB name
```
> You can also use a local `.env` file (ignored by git) and load it in `API.py` via `python-dotenv`.

### 1.3 Run the API
```powershell
uvicorn API:app --reload --port 8000
```
API base: `http://127.0.0.1:8000`

**Quick test**
```bash
curl "http://127.0.0.1:8000/complete?fragment=two%20things%20are%20infinite"
```

---

## 2) Step 2 — Client (ASR + Router + TTS)

> Uses Conda env `step2` with Python 3.11.

### 2.1 Create & activate
```bash
conda create -n step2 python=3.11 -y
conda activate step2
```

### 2.2 Install deps
```bash
pip install -r Step_2/requirements.txt
```

**`Step_2/requirements.txt`**
```
gradio==4.44.0
faster-whisper==1.0.3
numpy==1.26.4
soundfile==0.12.1
python-dotenv==1.0.1
requests==2.32.3
scikit-learn==1.5.1
webrtcvad==2.0.10
edge-tts==6.1.11
pyttsx3==2.90
pydub==0.25.1
```
> If you use **speechbrain ECAPA** for Speaker ID, also install: `speechbrain` (+ a compatible `torch`).

### 2.3 Set env vars for Step 2 (persist in conda)
```bash
conda env config vars set \
  STEP1_API_BASE="http://127.0.0.1:8000" \
  LLM_GGUF="C:\Users\shri\Data_Science\Text Mining\mistral-7b-instruct-v0.1.Q4_K_M.gguf" \
  ASR_MODEL=small \
  SIM_THRESHOLD=0.75
conda deactivate && conda activate step2
```

**Optional (demo only): enable LLM intent mapper — slower**
```bash
conda env config vars set USE_LLM_INTENTS=1
conda deactivate && conda activate step2
```

### 2.4 Run Step 2
```bash
cd Step_2
python main.py
```

---

## 🔄 End-to-End Order

1. Start **Neo4j** (database online and loaded).
2. Start **Step 1** API:
   ```powershell
   uvicorn API:app --reload --port 8000
   ```
3. Start **Step 2** client:
   ```bash
   conda activate step2
   cd Step_2
   python main.py
   ```
4. Interact; Step 2 calls Step 1 at `STEP1_API_BASE`.

---

## ⚙️ Key Environment Variables

| Variable            | Used by | Example                                      | Notes                              |
|---------------------|--------:|----------------------------------------------|------------------------------------|
| `NEO4J_URI`         |  API    | `neo4j://127.0.0.1:7687`                      | Bolt URI                           |
| `NEO4J_USER`        |  API    | `neo4j`                                       |                                    |
| `NEO4J_PASSWORD`    |  API    | *(set locally)*                               | **Never commit**                   |
| `NEO4J_DATABASE`    |  API    | `neo4j` or `quotes_db`                        | Must match your DB name            |
| `STEP1_API_BASE`    | Client  | `http://127.0.0.1:8000`                       | Where Step 2 calls Step 1          |
| `LLM_GGUF`          | Client  | `C:\…\mistral-7b-instruct-v0.1.Q4_K_M.gguf`   | Path to local GGUF (if used)       |
| `ASR_MODEL`         | Client  | `small`                                       | Faster-Whisper model size          |
| `SIM_THRESHOLD`     | Client  | `0.75`                                        | Rerank/ID similarity threshold     |
| `USE_LLM_INTENTS`   | Client  | `0` / `1`                                     | Optional; enables LLM router       |

---

## 🧪 Troubleshooting

- **Auth errors to Neo4j** → check `NEO4J_USER/NEO4J_PASSWORD`.
- **Database not found** → verify `NEO4J_DATABASE` (default is `neo4j`; custom could be `quotes_db`).
- **CORS from browser UI** → add CORS middleware in `API.py`.
- **Slow intent mapping** → keep `USE_LLM_INTENTS` unset/`0` (regex routing only).
- **Windows mic/ASR issues** → grant microphone permission; try launching terminal as admin once.

---

## 🔒 Security & Large Files

- Put secrets in env vars or a local `.env` (ignored by git).  
- Don’t commit files >100 MB (models/audio). Use **Git LFS** if you must share them.

---

## 📜 License & Acknowledgments

- Powered by Neo4j, Faster-Whisper, and (optionally) a local GGUF LLM.  
- Quotes were extracted from Wikiquote; please cite accordingly.

---

### 🐧 Linux/macOS equivalents (quick)

**Step 1**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-step1.txt
export NEO4J_URI="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<your-password>"
export NEO4J_DATABASE="neo4j"
uvicorn API:app --reload --port 8000
```

**Step 2 (one-shot env vars)**
```bash
export STEP1_API_BASE="http://127.0.0.1:8000"
export LLM_GGUF="/path/to/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
export ASR_MODEL=small
export SIM_THRESHOLD=0.75
python Step_2/main.py
```
