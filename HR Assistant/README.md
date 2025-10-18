# HR Assistant Chatbox

A local **Retrieval-Augmented Generation (RAG)** system built using **Streamlit**, **LangChain**, **Chroma**, and **Ollama**.  It lets you ask natural language questions about your own HR documents (Offline).

---

## Pipeline
1. `ingest.py` — Loads and chunks all PDFs in the `docs/` folder.  
2. `build_vector.py` — Creates a vector database (`data/chroma_db`) using MiniLM embeddings.  
3. `rag_app.py` — Streamlit interface that connects the Chroma DB with a local LLM (e.g., DeepSeek-R1).

---

## How to Use
1. Place your PDF documents in the `docs/` folder.  
2. Run the following commands:
   ```bash
   python ingest.py
   python build_vector.py
   streamlit run rag_app.py

---

## Requirement
1. Python 3.10+
2. Ollama installed
