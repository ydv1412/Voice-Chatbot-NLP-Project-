import os
from fastapi import FastAPI, Query
from neo4j import GraphDatabase

# ---- config (use env vars in prod) ----
URI  = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASS = os.getenv("NEO4J_PASSWORD", "shri@1412")
DB   = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(URI, auth=(USER, PASS))
app = FastAPI(title="Quotes Autocomplete API")

CYpher_AUTOCOMPLETE = """
CALL db.index.fulltext.queryNodes('quoteTextFT', $q + "*") YIELD node, score
OPTIONAL MATCH (node)-[:SAID_BY]->(speaker:Person)
OPTIONAL MATCH (node)-[:ABOUT]->(abt:Person)
OPTIONAL MATCH (node)-[:MISATTRIBUTED_TO]->(mis:Person)
OPTIONAL MATCH (node)-[:DISPUTED_WITH]->(disp:Person)
RETURN node.id AS id,
       node.text AS quote,
       node.source AS source,
       node.heading_context AS heading_context,
       node.status AS status,
       coalesce(speaker.name,'unknown') AS said_by,
       abt.name  AS about_person,
       mis.name  AS misattributed_to,
       disp.name AS disputed_with,
       score
ORDER BY score DESC
LIMIT $limit
"""

def _autocomplete(tx, q, limit):
    return [r.data() for r in tx.run(CYpher_AUTOCOMPLETE, q=q, limit=limit)]

@app.get("/autocomplete")
def autocomplete(q: str = Query(..., min_length=1), limit: int = 5):
    with driver.session(database=DB) as session:
        rows = session.execute_read(_autocomplete, q, limit)
    return {"count": len(rows), "results": rows}

@app.on_event("shutdown")
def _close():
    driver.close()
