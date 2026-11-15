import ollama

questions = [
    "Under the EU AI Act, what is the maximum fine for using a prohibited AI practice?",
    "What does Article 4 of the EU AI Act require?",
]

for q in questions:
    response = ollama.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": q}],
        think=False,
    )
    print("Q:", q)
    print("A:", response["message"]["content"])
    print("-" * 60)