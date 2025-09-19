
import os, re, tempfile, time
from pathlib import Path
from dotenv import load_dotenv
import soundfile as sf
import numpy as np

from audio_utils import record_utterance_wav
from asr import transcribe_file, transcribe_file_fast
from dialogue import handle_user_transcript
from tts import speak, list_voices, resolve_voice_id_and_name
from speaker_id import SpeakerID
from user_prefs import set_voice_prefs, get_prefs
from session import clear_session
from config import USE_SPK_ID, SPEAKER_DB_PATH, SPEAKER_ID_THRESHOLD, DEBUG

# optional (LLM intents)
USE_LLM_INTENTS = bool(int(os.getenv("USE_LLM_INTENTS", "1")))
if USE_LLM_INTENTS:
    from intent_mapper import map_intent_with_llm
else:
    def map_intent_with_llm(_text: str):
        return {"intent": "query", "slots": {}, "confidence": "low"}

# regex hints
COMMAND_HINT_RE = re.compile(
    r"\b(reset|clear|start over|register|enroll|my name is|"
    r"set|change|switch|voice|rate|speed|pace|tempo|volume|louder|quieter|softer|"
    r"test my voice|list voices|new quote|another quote)\b",
    re.I
)

QUOTEY_RE = re.compile(
    r"(?:\bquote\b|finish\b|complete\b|who said\b|who wrote\b|about whom\b|source\b|citation\b|disputed\b|misattributed\b|when\b)",
    re.I
)

# "find/get me a ..." not mentioning "quote"
SCOPE_GUARD_RE = re.compile(r'^\s*(find|get|search)\s+me\s+(?:a|the)\s+(?!quote\b).*', re.I)

# follow-up detector
FOLLOWUP_RE = re.compile(
    r"(?:\bwho said\b|\bwho wrote\b|\babout whom\b|\bsource\b|\bcitation\b|\bfinish\b|\bcomplete\b|\bwhen\b|\bdisputed\b|\bmisattributed\b)",
    re.I
)

# swithching the user
SELF_HINT_START_RE = re.compile(
    r"^\s*(?:this\s+is|i\s+am)\s+(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*(?:[.,!?])?\s*$"
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

# intent normalization
def _norm_intent_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r'[.!?…。،،]+$', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.lower()

# sticky speaker + session state
_LAST_SPK = None            # last recognized speaker name
_LAST_SPK_TS = 0.0          # when we last recognized them
ACTIVE_SESSION = "default"   # whose context we currently use

# tracking most recent recognition
_RECENT_RECOG_NAME = None
_RECENT_RECOG_TS = 0.0

# stick to last user for short utterance
STICKY_SHORT_SEC      = float(os.getenv("STICKY_SHORT_SEC", "0.9"))
STICKY_TTL_SEC        = float(os.getenv("STICKY_TTL_SEC", "18"))   # keep last speaker "active" for shorts
LONG_STICKY_TTL_SEC   = float(os.getenv("LONG_STICKY_TTL_SEC", "90"))  # only for follow-ups

# switch session
SWITCH_MIN_SEC        = float(os.getenv("SWITCH_MIN_SEC", "1.2"))   # long enough utterance can switch
SWITCH_MIN_SCORE      = float(os.getenv("SWITCH_MIN_SCORE", "0.66")) # if identify gives a score

# enrollment & name parsing
PENDING_ENROLL = False

REGISTER_ONLY_RE = re.compile(r'^\s*(register|enroll)(?:\s+me)?(?:\s+please)?\s*$', re.I)
REGISTER_RE      = re.compile(r'^\s*(?:register|enroll)\s+me\s+as\s+(?P<name>[\w .\'-]{2,})\s*$', re.I)
MY_NAME_IS_RE    = re.compile(r'^\s*my\s+name\s+is\s+(?P<name>[\w .\'-]{2,})\s*$', re.I)

NAME_TAIL_SPLIT  = re.compile(r'[,:;.!?]| and\b| but\b| please\b| register\b| enroll\b| set\b', re.I)

def _extract_name_from_text(raw: str) -> str | None:
    """
    Registration-only extractor.
    Accepts ONLY: 'my name is <Name>' (anywhere in the utterance).
    Does NOT accept 'I am ...' or 'This is ...' (too ambiguous in normal questions).
    """
    if not raw:
        return None
    m = re.search(r'\bmy\s+name\s+is\s+(?P<tail>.+)', raw, flags=re.I)
    if not m:
        return None
    tail = m.group("tail")
    tail = NAME_TAIL_SPLIT.split(tail, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", tail)
    if not tokens:
        return None
    name = " ".join(tokens[:4]).strip()
    return name if 2 <= len(name) <= 40 else None

def _explicit_self_hint_start(raw: str) -> str | None:
    """
    Session-switch hint ONLY when the ENTIRE utterance is a short self-intro:
    'This is John', 'I am John', optionally ending with a punctuation.
    Does not match when followed by more words/clause.
    """
    if not raw:
        return None
    m = SELF_HINT_START_RE.match(raw.strip())
    return m.group("name") if m else None

# deterministic intents
SET_VOICE_RE     = re.compile(r'^\s*set\s+my\s+(?:voice|audio|speaker)\s+to\s+(?P<voice>.+)\s*$', re.I)
LIST_VOICES_RE   = re.compile(r'^\s*(list|show)\s+voices\s*$', re.I)
TEST_VOICE_RE    = re.compile(r"^\s*(test|what('?s| is)\s+my\s+voice)\s*$", re.I)

SET_RATE_NUM_RE  = re.compile(r'^\s*set\s+my\s+(?:speaking\s+)?(?:rate|speed|pace|tempo)\s+to\s+(?P<num>\d{2,3})\s*$', re.I)
SET_RATE_WORD_RE = re.compile(
    r'^\s*(?:set|make)\s+my\s+(?:speaking\s+)?(?:rate|speed|pace|tempo)\s+to\s+'
    r'(?P<word>very\s+slow|slow|medium|fast|very\s+fast|low|high)\s*$',
    re.I
)
ADJUST_RATE_RE   = re.compile(r'^\s*(?:make|speak)\s+(?P<dir>faster|slower)\s*$', re.I)

NEW_QUOTE_RE     = re.compile(r"\b(new|another|different)\s+quote\b|\bfind\s+me\s+(?:a|another)\s+quote\b", re.I)

SET_VOLUME_NUM_RE  = re.compile(r'^\s*set\s+my\s+volume\s+to\s+(?P<num>(?:0(?:\.\d+)?|1(?:\.0+)?))\s*$', re.I)
SET_VOLUME_WORD_RE = re.compile(r'^\s*(?:set|make)\s+my\s+volume\s+to\s+(?P<word>mute|low|medium|normal|high|max|maximum)\s*$', re.I)
ADJUST_VOLUME_RE   = re.compile(r'^\s*(?:make|turn)\s+(?:it\s+)?(?P<dir>louder|quieter|softer)\s*$', re.I)

LOGOUT_RE        = re.compile(r'^\s*(logout|log\s*out|sign\s*out|bye|good\s*bye|goodnight|see\s*you)\b', re.I)
RESET_RE         = re.compile(r'^\s*(reset|clear\s*(?:context|it|this)?|start\s*over|new\s*session)\s*$', re.I)

SMALLTALK_RE     = re.compile(r'^\s*(hi|hello|hey|yo|thanks|thank you|ok|okay)\s*!?\s*$', re.I)

# speaking presets & helpers
RATE_PRESETS = {"very slow": 130, "slow": 150, "medium": 185, "fast": 210, "very fast": 240}
RATE_ALIASES = {"low": "slow", "high": "fast"}
VOLUME_PRESETS = {"mute": 0.0, "low": 0.6, "medium": 0.85, "normal": 0.85, "high": 1.0, "max": 1.0, "maximum": 1.0}
def _clamp(v, lo, hi): return max(lo, min(hi, v))

# enrollment phrases
ENROLL_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Common sense is a deposit of prejudices laid down before eighteen.",
    "Find me the quote Common sense is actually nothing more than a deposit of prejudices laid down in the mind prior to the age of eighteen.",
    "Has Albert Einstein said this quote, One of the sure signs of maturity is the ability to rise to the point of self criticism",
    "Numbers matter: three, five, seven, nine and forty two."
]

def _ensure_sid():
    thr = SPEAKER_ID_THRESHOLD if SPEAKER_ID_THRESHOLD else 0.62
    return SpeakerID(db_path=SPEAKER_DB_PATH, threshold=thr)

def _do_enrollment_flow(sid: SpeakerID, name: str) -> bool:
    speak(f"Okay {name}. We will read five short lines to register your voice.")
    for idx, sentence in enumerate(ENROLL_PROMPTS, 1):
        print(f"\nLine {idx}/5:\n» {sentence}")
        speak(f"Line {idx}. After the beep, please read this line.")
        speak(sentence)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, f"enroll_{idx}.wav")
            record_utterance_wav(p)
            try:
                sid.enroll(name, p)
            except Exception as e:
                print(f"[ERR] enrollment sample {idx} failed: {e}")
                speak("Sorry, that sample failed. Let's move to the next line.")
                continue
    speak(f"All set. I have registered you as {name}.")
    print(f"\nEnrolled '{name}' with {len(ENROLL_PROMPTS)} samples.")
    print(f"[INFO] Speaker DB: {(Path('.cache')/'speakers.json').resolve()}")
    # after enroll, make them the active session
    global ACTIVE_SESSION, _LAST_SPK, _LAST_SPK_TS
    ACTIVE_SESSION = name
    _LAST_SPK = name
    _LAST_SPK_TS = time.time()
    speak("You can continue with your conversation.")
    return True

# session switch helpers
def _explicit_self_hint_start(raw: str) -> str | None:
    """(redeclared below for clarity if imported elsewhere)"""
    if not raw:
        return None
    m = SELF_HINT_START_RE.match(raw.strip())
    return m.group("name") if m else None

def _maybe_switch_session(recognized_user, score, dur, raw_text):
    """Switch ACTIVE_SESSION only when confidence/conditions are met."""
    global ACTIVE_SESSION, _LAST_SPK, _LAST_SPK_TS

    if not recognized_user or recognized_user.lower() == "default":
        return  # don't switch to None/default

    if recognized_user == ACTIVE_SESSION:
        return

    ok_by_score = (score is not None) and (score >= SWITCH_MIN_SCORE)
    ok_by_dur   = dur >= SWITCH_MIN_SEC
    # STRICT: only accept 'This is John' / 'I am John' when the utterance is JUST that    
    hint_name   = _explicit_self_hint_start(raw_text or "")
    ok_by_hint  = bool(hint_name and hint_name.lower() in recognized_user.lower())

    if ok_by_score or ok_by_dur or ok_by_hint:
        if DEBUG:
            print(f"[SID] Switching session → {recognized_user} "
                  f"(score={score}, dur={dur:.2f}, hint={ok_by_hint})")
        ACTIVE_SESSION = recognized_user
        _LAST_SPK = recognized_user
        _LAST_SPK_TS = time.time()
    else:
        if DEBUG:
            print(f"[SID] Not switching (to {recognized_user}); "
                  f"need more speech or explicit 'This is {recognized_user}'.")

# intents
def _handle_system_intents(text: str, sid: SpeakerID | None) -> bool:
    global PENDING_ENROLL, ACTIVE_SESSION
    raw = text or ""
    t = _norm_intent_text(raw)
    uid = ACTIVE_SESSION  # use current active session for all intents
    if DEBUG: print(f"[DBG] INTENT_RAW={raw!r}  INTENT_NORM={t!r}  ACTIVE_UID={uid!r}")

    # small-talk
    if SMALLTALK_RE.match(t):
        speak("hello", user_id=uid)
        return True

    # reset / clear context 
    if RESET_RE.match(t):
        clear_session(uid)
        if DEBUG: print(f"[INFO] Cleared context for {uid!r}")
        speak("Context cleared. Ask me a new quote.", user_id=uid)
        return True

    # scope guard
    if SCOPE_GUARD_RE.match(t) and not QUOTEY_RE.search(t):
        speak("I can help with quotes. Say: 'find me the quote …' or 'finish the quote …'", user_id=uid)
        return True

    # pending enrollment gate
    if PENDING_ENROLL:
        name = _extract_name_from_text(raw)  # ONLY 'my name is ...'
        if not name:
            speak("I’m waiting for your name. Please say: 'My name is ...'", user_id=uid)
            return True
        PENDING_ENROLL = False
        if sid is None:
            speak("Speaker identification is disabled. Set USE_SPK_ID=1 to enable.", user_id=uid)
            return True
        if name in (sid.db.embeddings or {}):
            speak(f"{name} is already registered. I’ll add three more samples to your profile.", user_id=name)
        else:
            speak(f"Creating a new profile for {name}.", user_id=name)
        return _do_enrollment_flow(sid, name)

    # logout
    if LOGOUT_RE.match(t):
        clear_session(uid)
        print(f"[INFO] Logged out '{uid}'.")
        ACTIVE_SESSION = "default"
        speak("Logged out. I'll use the default voice until you register again.", user_id="default")
        return True

    # list/test voice
    if LIST_VOICES_RE.match(t):
        voices = list_voices()
        print("Available voices:")
        for vid, vname in voices:
            print(f"- id={vid!r}  name={vname!r}")
        sample = ", ".join([v[1] for v in voices[:5]]) or "none"
        speak(f"I found {len(voices)} system voices. For example: {sample}.", user_id=uid)
        return True

    if TEST_VOICE_RE.match(t):
        speak("This is your current voice setting. One, two, three.", user_id=uid)
        return True

    # set voice
    m = SET_VOICE_RE.search(t)
    if m:
        asked = m.group("voice").strip()
        vid, vname = resolve_voice_id_and_name(asked)
        if vid:
            prefs = set_voice_prefs(uid, voice=vid)
            print(f"Saved voice for '{uid}': {prefs}")
            print(f"[INFO] User prefs: {(Path('.cache')/'user_prefs.json').resolve()}")
            speak(f"Okay. I will use {vname} for {uid}.", user_id=uid)
        else:
            speak(f"I couldn't find a voice matching {asked}. Say 'list voices' to hear options.", user_id=uid)
        return True

    # rate/volume
    m = SET_RATE_NUM_RE.search(t)
    if m:
        rate = _clamp(int(m.group("num")), 120, 260)
        prefs = set_voice_prefs(uid, rate=rate)
        print(f"Saved rate for '{uid}': {prefs}")
        speak(f"Speaking rate set to {prefs['rate']} for {uid}.", user_id=uid)
        return True

    m = SET_RATE_WORD_RE.search(t)
    if m:
        key = RATE_ALIASES.get(m.group("word").lower(), m.group("word").lower())
        rate = RATE_PRESETS[key]
        prefs = set_voice_prefs(uid, rate=rate)
        print(f"Saved rate for '{uid}': {prefs}")
        speak(f"Okay. Rate set to {key} for {uid}.", user_id=uid)
        return True

    m = ADJUST_RATE_RE.search(t)
    if m:
        cur = get_prefs(uid)
        delta = 15 if m.group("dir").lower() == "faster" else -15
        rate = _clamp(int(cur.get("rate", 185)) + delta, 120, 260)
        prefs = set_voice_prefs(uid, rate=rate)
        speak(f"Done. New rate is {prefs['rate']} for {uid}.", user_id=uid)
        return True

    m = SET_VOLUME_NUM_RE.search(t)
    if m:
        vol = _clamp(float(m.group("num")), 0.0, 1.0)
        prefs = set_voice_prefs(uid, volume=vol)
        print(f"Saved volume for '{uid}': {prefs}")
        speak(f"Volume set to {prefs['volume']:.2f} for {uid}.", user_id=uid)
        return True

    m = SET_VOLUME_WORD_RE.search(t)
    if m:
        key = m.group("word").lower()
        vol = VOLUME_PRESETS[key]
        prefs = set_voice_prefs(uid, volume=vol)
        speak(f"Okay. Volume set to {key} for {uid}.", user_id=uid)
        return True

    m = ADJUST_VOLUME_RE.search(t)
    if m:
        cur = get_prefs(uid)
        delta = 0.10 if m.group("dir").lower() == "louder" else -0.10
        vol = _clamp(float(cur.get("volume", 1.0)) + delta, 0.0, 1.0)
        prefs = set_voice_prefs(uid, volume=vol)
        speak(f"Done. New volume is {prefs['volume']:.2f} for {uid}.", user_id=uid)
        return True

    # register by name (explicit) 
    m = REGISTER_RE.match(t)
    name = m.group("name").strip() if m else None

    if name:
        if sid is None:
            speak("Speaker identification is disabled. Set USE_SPK_ID=1 to enable.", user_id=uid)
            return True
        if name in (sid.db.embeddings or {}):
            speak(f"{name} is already registered. I’ll add three more samples to your profile.", user_id=name)
        else:
            speak(f"Creating a new profile for {name}.", user_id=name)
        return _do_enrollment_flow(sid, name)

    # bare register (arm enrollment; next turn expects 'my name is ...')
    if REGISTER_ONLY_RE.match(t) or ("register" in t and len(t) <= 30):
        if sid is None:
            speak("Speaker identification is disabled. Set USE_SPK_ID=1 to enable.", user_id=uid)
            return True
        # If we very recently recognized a user, allow adding samples directly
        now = time.time()
        if _RECENT_RECOG_NAME and (now - _RECENT_RECOG_TS) < 10.0:
            target = _RECENT_RECOG_NAME
            speak(f"I’ll add three more samples to {target}.", user_id=target)
            return _do_enrollment_flow(sid, target)
        PENDING_ENROLL = True
        speak("Okay. Please say: 'My name is …' with your full name.", user_id=uid)
        return True

    # new quote / another quote
    if NEW_QUOTE_RE.search(t):
        clear_session(uid)
        speak("Okay, fresh start. What's the new quote?", user_id=uid)
        return True

    # LLM fallback for fuzzy phrasing (optional) 
    if not USE_LLM_INTENTS:
        return False
    if not COMMAND_HINT_RE.search(t) or len(t.split()) < 2 or len(t) > 200:
        return False
    try:
        intent_obj = map_intent_with_llm(raw)
    except Exception as e:
        if DEBUG: print(f"[DBG] LLM intent mapping error: {e}")
        return False

    i = intent_obj.get("intent", "query")
    s = intent_obj.get("slots", {})
    if DEBUG: print(f"[DBG] LLM_INTENT={i}  SLOTS={s}")

    handled = False
    if i == "reset":
        clear_session(uid); speak("Context cleared. Ask me a new quote.", user_id=uid); handled = True
    elif i == "register":
        PENDING_ENROLL = True; speak("Say 'My name is …' and your full name to begin enrollment.", user_id=uid); handled = True
    elif i == "provide_name":
        name = s.get("name")
        if sid is None:
            speak("Speaker identification is disabled. Set USE_SPK_ID=1.", user_id=uid); handled = True
        elif name:
            if name in (sid.db.embeddings or {}):
                speak(f"{name} is already registered. I’ll add three more samples.", user_id=name)
            else:
                speak(f"Creating a new profile for {name}.", user_id=name)
            _do_enrollment_flow(sid, name); handled = True
        else:
            PENDING_ENROLL = True; speak("I didn't catch your name. Say 'My name is …' and your full name.", user_id=uid); handled = True
    elif i == "new_quote":
        clear_session(uid); speak("Okay, fresh start. What's the new quote?", user_id=uid); handled = True
    # (set_voice/rate/volume handled by regex already)

    return handled

# main loop 
def run_interactive():
    print("=== Voice Quotes Assistant ===")
    print("Press Enter to talk.")
    print("To register, say: 'register' (you'll read three short lines).")
    print("If you're already registered or don't want to register, just ask your quote question.")
    print("Examples: 'finish the quote two things are infinite' or 'who said this quote?'.")
    print("Say 'logout' or 'bye' to clear your session and use default TTS.  Ctrl+C to exit.")
    print("You can also say: 'set my voice to Zira/David',")
    print("  'set my rate/speed to slow/medium/fast' or 'faster/slower',")
    print("  'set my volume to low/medium/high' or 'make it louder/quieter'.")
    print("  say 'test my voice' to hear your current voice.")

    global sid, _LAST_SPK, _LAST_SPK_TS, ACTIVE_SESSION, _RECENT_RECOG_NAME, _RECENT_RECOG_TS
    sid = _ensure_sid() if USE_SPK_ID else None
    ACTIVE_SESSION = "default"

    while True:
        try:
            input("\n↩️  Press Enter to start recording...")
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "utt.wav")

                # Record mic → WAV
                record_utterance_wav(wav_path)

                # Identify speaker (optional) + stickiness for very short clips
                recognized_user, score = None, None
                dur = 0.0
                try:
                    data, sr = sf.read(wav_path)
                    if getattr(data, "ndim", 1) > 1:
                        data = np.mean(data, axis=1)
                    dur = float(len(data)) / float(sr or 16000)
                except Exception:
                    dur = 0.0

                if sid is not None:
                    try:
                        res = sid.identify(wav_path)  # name or (name, score)
                        if isinstance(res, tuple) and len(res) >= 2:
                            recognized_user, score = res[0], float(res[1])
                        else:
                            recognized_user = res
                            score = None
                    except Exception as e:
                        recognized_user = None
                        if DEBUG: print(f"[DBG] identify error: {e}")

                    now = time.time()
                    if recognized_user:
                        # update last-recognized & recent-recog trackers
                        _LAST_SPK, _LAST_SPK_TS = recognized_user, now
                        _RECENT_RECOG_NAME, _RECENT_RECOG_TS = recognized_user, now
                        if DEBUG: print(f"👤 Recognized: {recognized_user} (score={score})")
                    else:
                        # If super short clip, keep last speaker for shorts
                        if dur < STICKY_SHORT_SEC and _LAST_SPK and (now - _LAST_SPK_TS) < STICKY_TTL_SEC:
                            recognized_user = _LAST_SPK
                            if DEBUG: print(f"[SID] Short {dur:.2f}s → sticking to {_LAST_SPK}")
                        else:
                            if DEBUG: print(f"[SID] No ID (dur {dur:.2f}s); not sticking")

                # FAST ASR for command intents 
                text_fast = transcribe_file_fast(wav_path, language=None)
                if DEBUG: print(f"[DBG] FAST_ASR={text_fast!r}")

                # Decide if we should switch session BEFORE handling intents
                if recognized_user:
                    _maybe_switch_session(recognized_user, score, dur, text_fast or "")

                if text_fast and _handle_system_intents(text_fast, sid):
                    print(f"🗣️ You ({ACTIVE_SESSION}) [fast-intent]: {text_fast}")
                    continue

                # Full ASR for normal Q&A
                text = transcribe_file(wav_path, language=None)
                if not text:
                    print("ASR heard nothing.")
                    continue

                # Follow-up rescue: if looks like follow-up & no ID switch this turn,
                # temporarily stick to last speaker within LONG_STICKY_TTL_SEC
                t_norm = _norm_intent_text(text)
                if (recognized_user is None) and FOLLOWUP_RE.search(t_norm) and _LAST_SPK:
                    now = time.time()
                    if (now - _LAST_SPK_TS) < LONG_STICKY_TTL_SEC:
                        recognized_user = _LAST_SPK
                        if DEBUG:
                            print(f"[SID] Follow-up rescue → sticking to {_LAST_SPK} "
                                  f"(age={(now-_LAST_SPK_TS):.1f}s)")
                        # Ensure ACTIVE_SESSION follows this rescue for this turn
                        _maybe_switch_session(recognized_user, score, dur, text)

                print(f"🗣️ You ({ACTIVE_SESSION}): {text}")

                # Intents again on full text
                if _handle_system_intents(text, sid):
                    continue

                # Normal Q&A flow — always use ACTIVE_SESSION
                if DEBUG: print(f"[SID] Active session this turn: {ACTIVE_SESSION}")
                reply = handle_user_transcript(text, session_id=ACTIVE_SESSION)
                print(f"🤖 Bot:\n{reply}")

                # Speak with user's preferred voice
                if reply and reply.strip():
                    speak(reply, user_id=ACTIVE_SESSION)
                else:
                    print("[TTS] Nothing to speak (empty reply).")

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break

if __name__ == "__main__":
    run_interactive()
