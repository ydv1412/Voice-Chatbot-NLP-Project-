import json
from pathlib import Path
from typing import Optional, Dict, Any

APP_DIR = Path(__file__).resolve().parent
_PREFS_PATH = APP_DIR / ".cache" / "user_prefs.json"

_DEFAULTS = {
    "voice": "",     # engine voice id or name fragment
    "rate": 185,
    "volume": 1.0,
}

def _load() -> Dict[str, Dict[str, Any]]:
    if _PREFS_PATH.exists():
        return json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
    return {}

def _save(obj: Dict[str, Dict[str, Any]]) -> None:
    _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PREFS_PATH.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def get_prefs(user_id: str) -> Dict[str, Any]:
    data = _load()
    prefs = data.get(user_id) or {}
    return {**_DEFAULTS, **prefs}

def set_voice_prefs(user_id: str, voice: Optional[str] = None,
                    rate: Optional[int] = None, volume: Optional[float] = None) -> Dict[str, Any]:
    data = _load()
    cur = data.get(user_id) or {}
    if voice is not None:
        cur["voice"] = voice
    if rate is not None:
        cur["rate"] = int(rate)
    if volume is not None:
        cur["volume"] = float(volume)
    data[user_id] = cur
    _save(data)
    return {**_DEFAULTS, **cur}

def list_all() -> Dict[str, Dict[str, Any]]:
    return _load()
