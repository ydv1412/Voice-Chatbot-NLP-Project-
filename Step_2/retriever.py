
import re
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase
from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    NEO4J_FT_INDEX,
    DEBUG,
)

# stopwords kept minimal to preserve meaning
STOP = set(
    "a an and are as at be but by for from has have i in is it its of on or that the to was were will with you your we our this those these they them he she his her".split()
)

def _clean_tokens(s: str) -> List[str]:
    """Lowercase, strip punctuation (keep apostrophes), remove light stopwords."""
    s = re.sub(r'[\"“”]', "", s)
    s = re.sub(r"[^A-Za-z0-9' ]+", " ", s)
    toks = [t for t in s.lower().split() if t and t not in STOP]
    return toks

def _variants(fragment: str) -> List[str]:
    frag = (fragment or "").strip()
    toks = _clean_tokens(frag)
    joined = " ".join(toks)

    variants: List[str] = []
    if len(toks) >= 2:
        variants.append(f"\"{joined}\"")       # exact phrase
        variants.append(f"\"{joined}\"~3")     # phrase with slop
    if toks:
        variants.append(" AND ".join(toks))    # AND all tokens
        variants.append(" ".join(t + "*" for t in toks))  # wildcard tails

    # de-dup + drop empties
    seen, out = set(), []
    for q in variants:
        if q and q.strip() and q not in seen:
            out.append(q); seen.add(q)
    return out


# Note `people` is returned as a list of maps {rel, name}.
# Using a pattern comprehension avoids null placeholders when there are no matches.

_CYPHER = """
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
WITH node, score,
     [(node)-[r:SAID_BY|ABOUT|MISATTRIBUTED_TO|DISPUTED_WITH]->(p:Person)
       | {rel: type(r), name: p.name}] AS people
RETURN node.id AS id,
       node.text AS quote,
       node.source AS source,
       node.heading_context AS heading_context,
       node.status AS status,
       people,
       score
ORDER BY score DESC
LIMIT $limit
"""

class Retriever:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.db = NEO4J_DATABASE
        self.index = NEO4J_FT_INDEX

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        try:
            self.driver.close()
        except Exception:
            pass

    def _run_many(self, q: str, limit: int) -> List[Dict[str, Any]]:
        """Run the FT query with the given Lucene string and LIMIT."""
        with self.driver.session(database=self.db) as sess:
            return sess.run(
                _CYPHER,
                {"index": self.index, "q": q, "limit": limit},
            ).data()

    # simple, transparent re-ranker: token coverage + phrase bonus + normalized FT score
    def _score_candidate(self, fragment: str, cand: Dict[str, Any]) -> float:
        q_toks = set(_clean_tokens(fragment))
        c_toks = set(_clean_tokens(cand["quote"]))
        coverage = (len(q_toks & c_toks) / max(1, len(q_toks)))  # 0..1

        phrase_bonus = 1.0 if " ".join(_clean_tokens(fragment)) in " ".join(_clean_tokens(cand["quote"])) else 0.0

        # Lucene scores aren't bounded; lightly normalize with a cap
        try:
            score_norm = min(float(cand["score"]) / 10.0, 1.0)
        except Exception:
            score_norm = 0.0

        # weights: tune if needed
        return 0.55 * coverage + 0.35 * score_norm + 0.10 * phrase_bonus

    def search_topk(self, fragment: str, k: int = 5, min_score: float = 3.0, per_variant_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = per_variant_limit if per_variant_limit is not None else max(k, 5)

        pool: Dict[str, Dict[str, Any]] = {}
        for q in _variants(fragment):
            rows = self._run_many(q, limit=limit)
            if DEBUG:
                print(f"[DBG] FT_QUERY={q!r}  HITS={len(rows)}")
            for r in rows:
                try:
                    if float(r["score"]) < min_score:
                        continue
                except Exception:
                    continue

                rid = r["id"]
                if rid not in pool or float(r["score"]) > float(pool[rid]["score"]):
                    pool[rid] = r

            if len(pool) >= 3 * k:
                break

        # re-rank
        cands = list(pool.values())
        for c in cands:
            c["_rerank"] = self._score_candidate(fragment, c)

        cands.sort(key=lambda x: (x["_rerank"], x["score"]), reverse=True)
        return cands[:k]

    def search_best(self, fragment: str, min_score: float = 3.0) -> Optional[Dict[str, Any]]:
        """Convenience wrapper: return only the best candidate after rerank."""
        top = self.search_topk(fragment, k=5, min_score=min_score)
        return top[0] if top else None
