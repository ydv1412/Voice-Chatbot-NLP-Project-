from neo4j import GraphDatabase
import csv, random
import os

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASS = "shri@1412"
DB   = "neo4j"

driver = GraphDatabase.driver(URI, auth=(USER, PASS))

out_path = "C:/Users/shri/Data_Science/NLP/retrieval_gold.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

def sample_quotes(n=100):
    with driver.session(database=DB) as sess:
        rows = sess.run("MATCH (q:Quote) RETURN q.id AS id, q.text AS quote ORDER BY rand() LIMIT $n", n=n).data()
    return rows

rows = sample_quotes(100)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["fragment","gold_id"])
    for r in rows:
        quote = r["quote"]
        qid   = r["id"]
        words = quote.split()
        if len(words) < 5:  # skip too short
            continue
        # pick a random 4–6 word window as the fragment
        start = random.randint(0, max(0, len(words)-6))
        frag = " ".join(words[start:start+6])
        w.writerow([frag, qid])

driver.close()
