import re
import json
from pypdf import PdfReader

# 1. Read all pages into one string
reader = PdfReader("data/eu_ai_act.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)

# 2. NEW: a section starts at "Article 12" OR "ANNEX III" on its own line
parts = re.split(r"\n(?=(?:Article \d+|ANNEX [IVXL]+)\s*\n)", text)

# 3. Break long sections into overlapping pieces
def split_long(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# 4. NEW: label each section as an article number, an annex, or the preamble
def section_label(part):
    match = re.match(r"Article (\d+)", part)
    if match:
        return match.group(1)
    match = re.match(r"ANNEX ([IVXL]+)", part)
    if match:
        return "Annex " + match.group(1)
    return "preamble"

# 5. Build chunks with metadata
chunks = []
for part in parts:
    label = section_label(part)
    for piece in split_long(part):
        chunks.append({"id": len(chunks), "article": label, "text": piece})

# 6. Save
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=1)

annexes = sorted({c["article"] for c in chunks if c["article"].startswith("Annex")})
print(len(parts), "sections,", len(chunks), "chunks")
print("Annexes found:", annexes)
print("Article 113 chunks:", sum(1 for c in chunks if c["article"] == "113"))