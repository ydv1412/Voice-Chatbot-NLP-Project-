import threading, gc, re
import pyttsx3
from difflib import get_close_matches
from typing import Optional, Tuple
from user_prefs import get_prefs
from config import TTS_RATE, TTS_VOLUME, TTS_VOICE

_lock = threading.Lock()

def _norm_label(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'\b(microsoft|desktop|voice|english|united|states|us)\b', ' ', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _find_voice_id(engine: pyttsx3.Engine, wanted: str) -> Optional[str]:
    """Resolve a user-provided voice label (name/id/alias/typo) to an engine voice id."""
    if not wanted:
        return None
    voices = engine.getProperty("voices")
    w_raw = wanted.strip()
    w_low = w_raw.lower()
    w_norm = _norm_label(w_raw)

    # Exact id or name; raw substring
    for v in voices:
        vid, vname = getattr(v, "id", "") or "", getattr(v, "name", "") or ""
        if w_raw == vid or w_raw == vname:
            return vid
        if w_low in vid.lower() or w_low in vname.lower():
            return vid

    # Normalized substring + fuzzy
    labels, label_to_id = [], {}
    for v in voices:
        vid, vname = getattr(v, "id", "") or "", getattr(v, "name", "") or ""
        for lab in (vname, vid, _norm_label(vname), _norm_label(vid)):
            if lab:
                labels.append(lab)
                label_to_id.setdefault(lab, v.id)

    for lab in labels:
        if w_norm and w_norm in _norm_label(lab):
            return label_to_id[lab]

    if w_norm:
        best = get_close_matches(w_norm, labels, n=1, cutoff=0.6)
        if best:
            return label_to_id[best[0]]

    # Common ASR mis-hearings for "Zira"
    aliases = {"jira": "zira", "sira": "zira", "zera": "zira", "zero": "zira"}
    if w_norm in aliases:
        return _find_voice_id(engine, aliases[w_norm])

    return None

def resolve_voice_id_and_name(wanted: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (voice_id, friendly_name) for a wanted label, or (None, None)."""
    engine = pyttsx3.init(driverName="sapi5")
    try:
        vid = _find_voice_id(engine, wanted)
        vname = None
        if vid:
            for v in engine.getProperty("voices"):
                if v.id == vid:
                    vname = v.name
                    break
        return vid, vname
    finally:
        try: engine.stop()
        except Exception: pass

def speak(text: str, user_id: Optional[str] = None) -> None:
    """Speak text synchronously. If user_id is None, use global defaults."""
    if not text:
        return

    if user_id:
        prefs = get_prefs(user_id)
        voice = prefs.get("voice") or TTS_VOICE
        rate = int(prefs.get("rate") or TTS_RATE)
        volume = float(prefs.get("volume") or TTS_VOLUME)
    else:
        voice, rate, volume = TTS_VOICE, TTS_RATE, TTS_VOLUME

    with _lock:
        engine = pyttsx3.init(driverName="sapi5")
        try:
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            vid = _find_voice_id(engine, voice) if voice else None
            if vid:
                engine.setProperty("voice", vid)
                # to know which voice is actually used
                for v in engine.getProperty("voices"):
                    if v.id == vid:
                        print(f"[TTS] Using voice: {v.name}  (id={vid})  rate={rate}  vol={volume}")
                        break
            else:
                print(f"[TTS] No matching voice for {voice!r}; using system default.")
            engine.say(str(text))
            engine.runAndWait()
        finally:
            try: engine.stop()
            except Exception: pass
            del engine
            gc.collect()

def list_voices() -> list[tuple[str, str]]:
    engine = pyttsx3.init(driverName="sapi5")
    try:
        return [(v.id, v.name) for v in engine.getProperty("voices")]
    finally:
        try: engine.stop()
        except Exception: pass
