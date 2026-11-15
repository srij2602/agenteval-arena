import sys
import pandas as pd
import streamlit as st

sys.path.append("rag")
from retrieve import SEARCH_METHODS, load_index
from generate import generate_answer

st.set_page_config(page_title="AgentEval Arena", layout="wide")

@st.cache_resource            # load the index once, not on every click
def get_embeddings():
    return load_index()

embeddings = get_embeddings()

st.title("AgentEval Arena")
st.caption("RAG over the EU AI Act, built from scratch, with measured retrieval quality")

ask_tab, results_tab = st.tabs(["Ask a question", "Evaluation results"])

# ---------- Tab 1: ask ----------
with ask_tab:
    question = st.text_input("Your question", "What is the maximum fine for using a prohibited AI practice?")
    methods = [m for m in SEARCH_METHODS if m != "hyde"]   # HyDE is slow and didn't help
    col1, col2 = st.columns(2)
    method = col1.selectbox("Retrieval method", methods, index=methods.index("full"))
    k = col2.slider("Chunks to retrieve (k)", 1, 10, 5)
    use_llm = st.checkbox("Generate an answer with the local LLM (needs Ollama running)", value=True)

    if st.button("Search") and question:
        results = SEARCH_METHODS[method](question, embeddings, k)

        if use_llm:
            with st.spinner("Generating answer..."):
                text = generate_answer(question, results)
            if text:
                st.subheader("Answer")
                st.write(text)
            else:
                st.info("No LLM available here, so only the retrieved sources are shown.")

        st.subheader("Retrieved sources")
        for rank, (chunk, score) in enumerate(results, start=1):
            art = chunk["article"]
            label = art if art.startswith("Annex") else f"Article {art}"
            with st.expander(f"{rank}. {label}  (score {score:.3f})"):
                st.text(chunk["text"])

# ---------- Tab 2: results ----------
with results_tab:
    df = pd.read_csv("results/retrieval_results.csv")
    st.subheader("Retrieval methods compared (18 golden questions)")
    st.dataframe(df, hide_index=True)
    # st.bar_chart(df.set_index("method")[["Hit@5", "MRR"]])
    st.bar_chart(df.set_index("method")[["Hit@5", "MRR"]], stack=False)    
    st.markdown("""
**Key findings**
- Excluding the preamble was the biggest single gain (Hit@1 0.44 -> 0.72).
- Metadata routing fixed "Article 4", which no search method could find.
- Reranking gave the best Hit@5 (0.94), at the cost of fewer #1 hits.
- HyDE and BM25 did not help on legal text.
""")