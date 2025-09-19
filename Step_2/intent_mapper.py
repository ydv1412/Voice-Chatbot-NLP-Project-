# optional LLM Intent mapper. If want to try set the USE_LLM_INTENTS =1 in the main.py
from typing import Dict, Any
import json
from llm import LLM  
_llm = LLM()        

INTENT_PROMPT = (
    'You are an intent mapper. Return ONLY compact JSON with this schema: '
    '{"intent":"<one of: reset, register, provide_name, set_voice, list_voices, test_voice, '
    'set_rate, bump_rate, set_volume, bump_volume, new_quote, query, smalltalk>", '
    '"slots":{}, "confidence":"low|medium|high"}\n'
    'Rules:\n'
    '- Never include explanations, code fences, or extra text.\n'
    '- "my name is X" => provide_name, slots={"name": X}.\n'
    '- Voice change => set_voice, slots={"voice": <name>} (normalize "jira"->"Zira").\n'
    '- Speed/rate/pace/tempo => set_rate; accept words (slow/fast/very fast) or numbers 80–300.\n'
    '- faster/slower => bump_rate {"direction":"faster|slower"}.\n'
    '- louder/quieter/softer => bump_volume {"direction":"louder|quieter|softer"}.\n'
    '- "new quote", "another quote" => new_quote.\n'
    '- "reset", "clear context", "start over" => reset.\n'
    '- Otherwise use query unless it is pure smalltalk.\n'
    'Respond with JSON only.'
)

def map_intent_with_llm(text: str) -> Dict[str, Any]:
    raw = _llm.chat(text=f'User: "{text}"', system=INTENT_PROMPT)
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("non-dict")
    except Exception:
        return {"intent": "query", "slots": {}, "confidence": "low"}

    intent = (obj.get("intent") or "query").strip()
    slots = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    conf = obj.get("confidence") if obj.get("confidence") in ("low", "medium", "high") else "low"
    return {"intent": intent, "slots": slots, "confidence": conf}
