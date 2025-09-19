import os, re, json
from typing import List, Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv
from config import DEBUG

load_dotenv(override=True)

MODEL_PATH = os.getenv("LLM_GGUF")
if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    raise RuntimeError("Set LLM_GGUF to your .gguf path")

MISTRAL_INSTRUCT_TEMPLATE = r"""{{ bos_token }}{% for message in messages %}
{% if message['role'] == 'system' %}[INST] <<SYS>>
{{ message['content'] }}
<</SYS>>{% elif message['role'] == 'user' %}[INST]
{{ message['content'] }}{% elif message['role'] == 'assistant' %}
{{ message['content'] }}[/INST]{% endif %}{% endfor %}"""

# extractor (3 shot) 
SYSTEM_EXTRACT = (
  "You are a precise span extractor. Return ONLY JSON: {\"fragment\": string}.\n"
  "Goal: extract ONLY the quote fragment to search for (not commands or author names).\n"
  "Remove command words: new quote, another quote, different quote, find, search, look up, "
  "complete, finish, continue, tell me, who said, source.\n"
  "Do NOT include author names or sources. Do NOT include trailing punctuation.\n"
  "If you cannot find at least 3 meaningful words of the quote, return {\"fragment\": \"\"}.\n"
  "\nExamples:\n"
  "Q: Find the quote: Albert Einstein was almost considered as a superhuman.\n"
  "A: {\"fragment\": \"Albert Einstein was almost considered as a superhuman\"}\n"
  "Q: Complete this new quote as an eminent pioneer in the realm of high.\n"
  "A: {\"fragment\": \"as an eminent pioneer in the realm of high\"}\n"
  "Q: Who said this? \"Two things are infinite...\"\n"
  "A: {\"fragment\": \"Two things are infinite\"}\n"
  "No extra text."
)

SYSTEM_ANSWER = (
  "You are a quotes assistant. You will be given the user's question and a list of database results.\n"
  "Write a concise answer USING ONLY the provided results. If results are empty, say you couldn't find it.\n"
  "If multiple matches exist, show the best 1–3 with author and source. Do not invent authors or sources."
)

SYSTEM_REQUEST_FIELDS = (
  "You map a user question about a quote into a list of requested fields.\n"
  "Return ONLY JSON like: {\"fields\": [ ... ]}\n"
  "Valid fields:\n"
  "  - \"said_by\"\n  - \"about_person\"\n  - \"misattributed_to\"\n  - \"disputed_with\"\n  - \"source\"\n  - \"finish_quote\"\n  - \"when\"\n"
  "Rules:\n1) Use plural intent if user asks for more than one thing.\n2) If unclear, return an empty list.\nNo extra text."
)

SYSTEM_DECIDE = (
  "Decide the next action for a quotes assistant.\n"
  "Return ONLY JSON:{\"action\": one_of[\"answer_from_memory\",\"search_db\",\"clarify\",\"reset\"], \"query\": string}\n"
  "Definitions:\n- answer_from_memory: user asks about the SAME quote and we already have context.\n"
  "- search_db: user mentions a NEW quote or gives a fragment; provide a minimal search query.\n"
  "- clarify: the request is unclear; ask the user to provide a quote fragment.\n"
  "- reset: user asks to reset/clear/start over.\n"
  "Rules:\n1) has_context=true means we already have the last quote facts.\n"
  "2) If user says reset/clear/start over -> action=reset.\n"
  "3) If has_context=true AND the user asks follow-ups (who/about/source/finish/disputed/when) possibly with 'this/that/the quote/it' -> action=answer_from_memory.\n"
  "4) If has_context=false and the user asks a follow-up like 'who said this?' with no quote given -> action=clarify.\n"
  "5) If the user provides a quote fragment or asks to find/complete a specific quote -> action=search_db with a minimal query (only the fragment words).\n"
  "6) No extra text."
)

def _json_only(s: str) -> dict:
    for m in re.finditer(r"\{.*?\}", s, flags=re.S):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}

from llama_cpp import Llama  
_llama = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=max(1, (os.cpu_count() or 8) - 2),
    n_batch=256,
    verbose=False,
    chat_template=MISTRAL_INSTRUCT_TEMPLATE,
    seed=0,
)

@dataclass
class ChatState:
    history: List[Dict[str, str]] = field(default_factory=list)
    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
    def clear(self) -> None:
        self.history.clear()

def _as_messages(system: str, user: str, history: List[Dict[str,str]] | None = None):
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs += history
    msgs.append({"role": "user", "content": user})
    return msgs

# noise words to strip after extraction
NOISE_RE = re.compile(
    r'\b(new|another|different)\s+quote\b|'
    r'\b(find|search|look up|complete|finish|continue|tell me|who said|source)\b[: ]?',
    re.I,
)

class LLM:
    def __init__(self):
        self.chat_state = ChatState()

    def _chat_complete(self, system: str, user: str, max_tokens: int = 128) -> str:
        out = _llama.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    @staticmethod
    def _clean_answer(s: str) -> str:
        s = s.strip()
        junk_starts = (
            "understood", "sure", "okay", "i will", "i can", "as an ai",
            "here is", "here's", "let me", "i’ll", "i will provide"
        )
        low = s.lower()
        for j in junk_starts:
            if low.startswith(j):
                s = s.split("\n", 1)[-1] if "\n" in s else ""
                s = s.lstrip(",:-. ").strip()
                break
        return s

    # Routers
    def extract_fragment(self, question: str) -> str:
        qlow = (question or "").lower()
        if any(phrase in qlow for phrase in ["who said this", "who said that", "who said it", "who wrote this"]):    #follow-sup
            if DEBUG: 
                print(f"[DBG] FRAGMENT skipped for follow-up: {question!r}")
            return ""

        out = _llama.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_EXTRACT},
                      {"role": "user", "content": question}],
            temperature=0.0, max_tokens=64,
        )
        data = _json_only(out["choices"][0]["message"]["content"])
        frag = (data.get("fragment") or "").strip()
        frag = frag.strip('“”"\' .,:;!?-')        # strip punct
        frag = NOISE_RE.sub("", frag).strip('“”"\' .,:;!?-')     # strip command/noise words
        tokens = re.findall(r"\w+", frag)
        frag = frag if len(tokens) >= 3 else ""                 # require a minimum frag len >=3
        if DEBUG: print(f"[DBG] FRAGMENT={frag!r}")
        return frag

    def extract_requested_fields(self, question: str) -> list[str]:
        out = _llama.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_REQUEST_FIELDS},
                      {"role": "user", "content": question}],
            temperature=0.0, max_tokens=64,
        )
        data = _json_only(out["choices"][0]["message"]["content"])
        fields = data.get("fields") or []
        allowed = {"said_by","about_person","misattributed_to",
                   "disputed_with","source","finish_quote","when"}
        return [f for f in (x.strip() for x in fields) if f in allowed]

    def decide_action(self, user_text: str, has_context: bool) -> dict:
        prompt = f"has_context={str(has_context).lower()}\nuser: {user_text}"
        out = _llama.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_DECIDE},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=96,
        )
        data = _json_only(out["choices"][0]["message"]["content"])
        action = (data.get("action") or "").strip()
        if action not in {"answer_from_memory","search_db","clarify","reset"}:
            action = "clarify"
        query = (data.get("query") or "").strip()
        if DEBUG: print(f"[DBG] DECIDE_ACTION has_context={has_context} action={action} query={query!r}")
        return {"action": action, "query": query}

    # Draft Response
    def answer_from_fields(self, question: str, facts: dict, fields: list[str], labeled: bool = True) -> str:
        people = facts.get("people") or []
        quote = (facts.get("quote") or "").strip()
        source = (facts.get("source") or "").strip()

        def _names(rel: str) -> list[str]:
            return [p["name"] for p in people if p.get("rel") == rel and p.get("name")]

        def _join(names: list[str]) -> str:
            if not names: return ""
            return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]

        def _when() -> str:
            text = " ".join(filter(None, [facts.get("heading_context"), source]))
            m = re.findall(r"\b(1[6-9]\d{2}|20\d{2})\b", text or "")
            return m[0] if m else ""

        LABEL = {
            "said_by": "Said by",
            "about_person": "About",
            "misattributed_to": "Misattributed to",
            "disputed_with": "Disputed with",
            "source": "Source",
            "finish_quote": "Quote",
            "when": "When",
        }

        if "finish_quote" in fields and len(fields) == 1:
            return quote

        parts: list[str] = []
        for f in fields:
            if f == "said_by":
                text = _join(_names("SAID_BY")) or "Unknown"
            elif f == "about_person":
                text = _join(_names("ABOUT")) or "Unknown"
            elif f == "misattributed_to":
                text = _join(_names("MISATTRIBUTED_TO")) or "None recorded"
            elif f == "disputed_with":
                text = _join(_names("DISPUTED_WITH")) or "No disputes recorded"
            elif f == "source":
                text = source or "Unknown"
            elif f == "finish_quote":
                text = quote or ""
            elif f == "when":
                text = _when() or "Unknown date"
            else:
                continue

            if labeled and f != "finish_quote":
                parts.append(f"{LABEL.get(f, f)}: {text}.")
            else:
                if text and not text.endswith(('.', '!', '?', '”')):
                    text += "."
                parts.append(text)

        return " ".join(p for p in parts if p)

    def answer_from_data(self, question: str, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results[:5], 1):
            q = (r.get("quote") or "").replace("\n", " ").strip()
            a = (r.get("author") or "").strip()
            s = (r.get("source") or "").strip()
            hc = (r.get("heading_context") or "").strip()
            lines.append(f"{i}. quote={q!r}; author={a!r}; source={s!r}; context={hc!r}")
        ctx = "DB_RESULTS:\n" + ("\n".join(lines) if lines else "(none)")

        prompt = (
            f"USER_QUESTION:\n{question}\n\n"
            f"{ctx}\n\n"
            "Instructions:\n"
            "- If DB_RESULTS is (none), say you couldn't find it.\n"
            "- Otherwise, answer using ONLY DB_RESULTS. Prefer the best matching item.\n"
            "- Return a short, natural answer (1–3 sentences). Include author and source when present."
        )

        out = _llama.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM_ANSWER},
                      {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=220,
        )
        return out["choices"][0]["message"]["content"].strip()

    def answer_from_facts(self, question: str, facts: dict) -> str:
        import re

        q = (question or "").strip().lower()

        quote  = (facts.get("quote") or "").strip()
        src    = (facts.get("source") or "").strip()
        people = facts.get("people") or []

        said_by = [p["name"] for p in people if p.get("rel") == "SAID_BY" and p.get("name")]
        about_p = [p["name"] for p in people if p.get("rel") == "ABOUT" and p.get("name")]
        misatt  = [p["name"] for p in people if p.get("rel") == "MISATTRIBUTED_TO" and p.get("name")]
        disput  = [p["name"] for p in people if p.get("rel") == "DISPUTED_WITH" and p.get("name")]

        def _join(names):
            names = [n for n in names if n]
            if not names:
                return ""
            if len(names) == 1:
                return names[0]
            return ", ".join(names[:-1]) + " and " + names[-1]

        # Intent flags
        is_who      = any(k in q for k in ["who said", "who wrote", "author", "who is the author"])
        is_about    = any(k in q for k in ["who is it about", "about whom", "about who", "who is this about"])
        is_source   = any(k in q for k in ["source", "citation", "reference", "where is this from", "origin"])
        is_finish   = any(k in q for k in ["finish the quote", "complete the quote", "finish quote", "complete this quote", "continue the quote"])
        is_disputed = any(k in q for k in ["disputed", "dispute", "contested", "is it true", "misattributed"])

        # Follow-ups
        if is_who:
            if said_by:
                return _join(said_by)
            if misatt:
                return f"Unknown; often misattributed to {_join(misatt)}."
            return "Unknown."

        if is_about:
            return _join(about_p) if about_p else "Unknown."

        if is_disputed:
            if disput and misatt:
                return f"Yes — disputed with {_join(disput)} and often misattributed to {_join(misatt)}."
            if disput:
                return f"Yes — disputed with {_join(disput)}."
            if misatt:
                return f"Yes — often misattributed to {_join(misatt)}."
            return "No disputes recorded."

        if is_source:
            return src or "Unknown source."

        if is_finish:
            return quote

        
        def _clean_qtext(s: str) -> str:
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"\s*\.\s*\.\s*", ". ", s)
            return s

        full = _clean_qtext(quote)
        author = _join(said_by)
        if full:
            if author and src:
                return f"\"{full}\" - {author}. {src}"
            if author:
                return f"\"{full}\" - {author}."
            if src:
                return f"\"{full}\"  {src}"
            return f"\"{full}\""

        # LLM rephrase
        rel_lines = [f"- {p.get('rel')}: {p.get('name')}" for p in people if p.get("name")]
        rel_block = "\n".join(rel_lines) if rel_lines else "(none)"
        system = (
            "You are a quotes assistant.\n"
            "Answer ONLY from the facts provided below.\n"
            "Be concise but do NOT invent missing text.\n"
            "If there is a quote text, prefer outputting the quote verbatim."
        )
        context = f"Quote: {quote}\nSource: {src}\nConnections:\n{rel_block}\n"
        user = f"{context}\nUser question: {question}\nYour answer:"
        raw = self._chat_complete(system, user, max_tokens=128)
        return self._clean_answer(raw)

    # ---------- Simple chat ----------
    def chat(self, text: str, system: str = "You are helpful.") -> str:
        out = _llama.create_chat_completion(
            messages=_as_messages(system, text, self.chat_state.history),
            temperature=0.6,
            max_tokens=256,
        )
        reply = out["choices"][0]["message"]["content"].strip()
        self.chat_state.add("user", text); self.chat_state.add("assistant", reply)
        return reply
