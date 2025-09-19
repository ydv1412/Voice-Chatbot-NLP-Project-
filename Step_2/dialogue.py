import re
from typing import Optional
from config import DEBUG
from session import get_session
from retriever import Retriever
from llm import LLM

_llm = LLM()
_ret = Retriever()

NEW_QUOTE_RE = re.compile(r'\b(new|another|different)\s+quote\b|\bfind\s+me\s+(?:a|another)\s+quote\b', re.I)
SMALLTALK_RE = re.compile(r'^(thanks|thank you|ok|okay|hmm|huh|great|nice)\.?$', re.I)

def handle_user_transcript(transcript: str, session_id: str = "default") -> str:
    text = (transcript or "").strip()
    sess = get_session(session_id)
    has_context = bool(sess.last_facts)

    if DEBUG:
        print(f"[DBG] USER_TEXT={text!r}  has_context={has_context}  session={session_id!r}")

    # short acknowledgement
    if SMALLTALK_RE.fullmatch(text):
        return "Thank You"

    # new/another quote clears context for a user
    if NEW_QUOTE_RE.search(text):
        sess.last_facts = None
        has_context = False
        if DEBUG:
            print("[DBG] NEW_QUOTE -> cleared session context")

    # extract quote fragment 
    fragment = _llm.extract_fragment(text) or ""
    if DEBUG:
        print(f"[DBG] FRAGMENT={fragment!r}")

    if fragment:
        row = _ret.search_best(fragment)
        if row:
            sess.last_facts = row
            return _llm.answer_from_facts(text, row)
        else:
            return "I couldn't find that exact quote."

    # no new fragment; if we have context, answer from memory
    if has_context and sess.last_facts:
        return _llm.answer_from_facts(text, sess.last_facts)

    # otherwise
    return "Give me a few words from the quote you have in mind."
