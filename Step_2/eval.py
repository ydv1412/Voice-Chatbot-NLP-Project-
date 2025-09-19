# retrieval evaluation
import csv, time, statistics
from retriever import Retriever

TOPK = 5
IN   = "C:/Users/shri/Data_Science/NLP/test_dataset/fragments_id.csv"
# OUT  = "retrieval_eval_results.csv"

ret = Retriever()
def rank_of(gold_id, results):
    for i, r in enumerate(results, 1):
        if (r.get("id") or "").strip() == gold_id.strip():
            return i
    return None

lat, n = [], 0
top1 = top5 = mrr5 = 0.0
rows_out = []

with open(IN, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        frag = row["fragment"].strip(); gold = row["gold_id"].strip()
        n += 1
        t0 = time.perf_counter()
        results = ret.search_topk(frag, k=TOPK)  
        dt = time.perf_counter() - t0; lat.append(dt)
        rank = rank_of(gold, results)
        top1 += 1 if rank == 1 else 0
        top5 += 1 if (rank and rank <= TOPK) else 0
        mrr5 += (1.0 / rank) if (rank and rank <= TOPK) else 0.0
        rows_out.append({"fragment": frag, "gold_id": gold, "rank": rank or "", "latency_s": f"{dt:.3f}"})

ret.close()

acc1 = top1/max(1,n); rec5 = top5/max(1,n); mrr = mrr5/max(1,n)
p50 = statistics.median(lat) if lat else 0.0
p90 = statistics.quantiles(lat, n=10)[8] if len(lat) >= 10 else p50
print(f"Samples {n} | Top1 {acc1:.3f} | Top5 {rec5:.3f} | MRR@5 {mrr:.3f} | p50 {p50:.3f}s p90 {p90:.3f}s")
