import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

ASR_MODEL = os.getenv("ASR_MODEL", "small")
ASR_MODEL_FAST = os.getenv("ASR_MODEL_FAST", "tiny")

_full_model = None
_fast_model = None

def _get_full_model():
    global _full_model
    if _full_model is None:
        import whisper
        _full_model = whisper.load_model(ASR_MODEL)
    return _full_model

def _get_fast_model():
    global _fast_model
    if _fast_model is None:
        import whisper
        _fast_model = whisper.load_model(ASR_MODEL_FAST)
    return _fast_model

def transcribe_file(path: str, language: Optional[str] = None) -> str:
    model = _get_full_model()
    out = model.transcribe(
        path,
        language=language,
        condition_on_previous_text=False,
        temperature=0.0,
        fp16=False,
        without_timestamps=True,
        logprob_threshold=-1.0,
        no_speech_threshold=0.5,
    )
    return (out.get("text") or "").strip()

def transcribe_file_fast(path: str, language: Optional[str] = None) -> str:
    model = _get_fast_model()
    out = model.transcribe(
        path,
        language=language,
        condition_on_previous_text=False,
        temperature=0.0,
        fp16=False,
        without_timestamps=True,
        logprob_threshold=-1.0,
        no_speech_threshold=0.5,
    )
    return (out.get("text") or "").strip()
