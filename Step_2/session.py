from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Session:
    last_facts: Optional[Dict[str, Any]] = None

_SESSIONS: Dict[str, Session] = {}

def get_session(user_id: str = "default") -> Session:
    if user_id not in _SESSIONS:
        _SESSIONS[user_id] = Session()
    return _SESSIONS[user_id]

def clear_session(user_id: str = "default") -> None:
    if user_id in _SESSIONS:
        _SESSIONS[user_id].last_facts = None

def clear_all_sessions() -> None:
    _SESSIONS.clear()
