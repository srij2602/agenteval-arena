import sys
import json
import time
sys.path.append("rag")

from metrics import first_rank, hit_at_k, mrr
from retrieve import dense_search, hyde_search, bm25_search, hybrid_search, load_index
from retrieve import SEARCH_METHODS, load_index

# Choose the method from the command line: dense (default) or hyde
method = sys.argv[1] if len(sys.argv) > 1 else "dense"
# search = {"dense": dense_search, "hyde": hyde_search, "bm25": bm25_search, "hybrid": hybrid_search}[method]
search = SEARCH_METHODS[method]

embeddings = load_index()
with open("data/golden_set.json", encoding="utf-8") as f:
    golden = json.load(f)

answerable = [g for g in golden if g["article"] is not None]

ranks = []
start = time.time()
for g in answerable:
    results = search(g["question"], embeddings, k=50)
    retrieved = [c["article"] for c, _ in results]
    rank = first_rank(retrieved, g["article"])
    ranks.append(rank)
    status = "OK  " if rank is not None and rank <= 5 else "MISS"
    print(f"{status} rank={rank} | Article {g['article']} | {g['question']}")
seconds_per_query = (time.time() - start) / len(answerable)

print(f"\n=== {method} retrieval ===")
for k in [1, 3, 5, 10]:
    print(f"Hit@{k}: {hit_at_k(ranks, k):.2f}")
print(f"MRR:   {mrr(ranks):.2f}")
print(f"Avg time per query: {seconds_per_query:.2f} s")