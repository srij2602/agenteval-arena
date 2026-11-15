import json
import re
import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open("data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# True for Articles and Annexes (binding rules), False for the preamble (recitals)
is_rule = np.array([c["article"] != "preamble" for c in chunks])

def build_index():
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save("data/embeddings.npy", embeddings)

def load_index():
    return np.load("data/embeddings.npy")

# ---------- Dense (embeddings) ----------
def dense_search(query, embeddings, k=5, skip_preamble=False):
    q = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ q                               # cosine similarity
    if skip_preamble:
        scores = np.where(is_rule, scores, -np.inf)       # preamble can never win
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]

# ---------- HyDE ----------
HYDE_PROMPT = """Write a short passage (3-4 sentences) in the formal style of the EU AI Act that answers this question.
Write only the passage, no explanation.

Question: {question}
Passage:"""

def hyde_search(query, embeddings, k=5):
    response = ollama.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=query)}],
    )
    passage = response["message"]["content"].split("</think>")[-1].strip()
    return dense_search(passage, embeddings, k)

# ---------- BM25 (keywords) ----------
def tokenize(text):
    return re.findall(r"\w+", text.lower())

bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

def bm25_search(query, embeddings, k=5, skip_preamble=False):
    scores = bm25.get_scores(tokenize(query))
    if skip_preamble:
        scores = np.where(is_rule, scores, -np.inf)
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]

# ---------- Hybrid (dense + BM25 with Reciprocal Rank Fusion) ----------
def hybrid_search(query, embeddings, k=5, skip_preamble=False, rrf_k=60):
    pool = max(k, 50)
    dense = dense_search(query, embeddings, pool, skip_preamble)
    sparse = bm25_search(query, embeddings, pool, skip_preamble)
    scores = {}
    for results in (dense, sparse):
        for rank, (chunk, _) in enumerate(results, start=1):
            scores[chunk["id"]] = scores.get(chunk["id"], 0) + 1 / (rrf_k + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(chunks[i], scores[i]) for i in best]

# ---------- NEW: Metadata routing ----------
def route_by_article(query, results, k=5):
    """If the question names 'Article N', put that article's chunks first."""
    named = set(re.findall(r"\barticle\s+(\d+)", query.lower()))
    if not named:
        return results[:k]
    pinned = [(c, 1.0) for c in chunks if c["article"] in named]
    rest = [(c, s) for c, s in results if c["article"] not in named]
    return (pinned + rest)[:k]

# ---------- NEW: Reranker (cross-encoder) ----------
reranker = None

def get_reranker():
    """Load the reranker only the first time it's needed (lazy loading)."""
    global reranker
    if reranker is None:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker

def rerank_search(query, embeddings, k=5, pool=30):
    candidates = dense_search(query, embeddings, pool, skip_preamble=True)   # fast first round
    pairs = [(query, c["text"]) for c, _ in candidates]
    scores = get_reranker().predict(pairs)                                   # careful judge
    order = np.argsort(-scores)
    return [(candidates[i][0], float(scores[i])) for i in order][:k]

# ---------- Combinations used in the evaluation ----------
def dense_filtered(query, embeddings, k=5):
    return dense_search(query, embeddings, k, skip_preamble=True)

def hybrid_filtered(query, embeddings, k=5):
    return hybrid_search(query, embeddings, k, skip_preamble=True)

def routed_search(query, embeddings, k=5):
    return route_by_article(query, dense_filtered(query, embeddings, k), k)

def full_search(query, embeddings, k=5):
    return route_by_article(query, rerank_search(query, embeddings, k), k)

SEARCH_METHODS = {
    "dense": dense_search,
    "hyde": hyde_search,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
    "dense_filtered": dense_filtered,
    "hybrid_filtered": hybrid_filtered,
    "routed": routed_search,
    "rerank": rerank_search,
    "full": full_search,
}

if __name__ == "__main__":
    build_index()
    embeddings = load_index()
    for chunk, score in dense_search("What is the maximum fine for prohibited AI practices?", embeddings):
        preview = chunk["text"][:100].replace("\n", " ")
        print(round(score, 3), "| Article", chunk["article"], "|", preview)