import numpy as np
from retrieve import model, chunks, load_index

embeddings = load_index()

# 1. Do the chunks exist?
for art in ["4", "99"]:
    found = [c for c in chunks if c["article"] == art]
    print(f"Article {art}: {len(found)} chunks")
    if found:
        print("   starts with:", found[0]["text"][:120].replace("\n", " "))

# 2. At what rank does the correct article first appear?
tests = [
    ("Under the EU AI Act, what is the maximum fine for using a prohibited AI practice?", "99"),
    ("What does Article 4 of the EU AI Act require?", "4"),
]
for question, art in tests:
    q = model.encode(question, normalize_embeddings=True)
    order = np.argsort(-(embeddings @ q))
    rank = next((r for r, i in enumerate(order, start=1) if chunks[i]["article"] == art), None)
    print(f"Correct Article {art} first appears at rank {rank}")