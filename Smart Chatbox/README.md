# Smart Chatbox

A local **document-aware AI assistant** built using **Streamlit, LangChain, Chroma**, and **Ollama**.  
It lets you upload PDFs, TXT, or DOCX files and chat with the content using local LLMs like **DeepSeek-R1**, **Mistral**, or **Llama 3** — all with conversational memory.

Features:
- Multi-model support (DeepSeek, Mistral, Llama 3)
- Conversational memory
- Adjustable temperature and token limits
- Sidebar for easy parameter control
- Works fully offline

---

## Pipeline

1. **Upload Files** — Add PDF, TXT, or DOCX files directly in the app.  
2. **Vectorization** — Documents are chunked and embedded using MiniLM and stored in Chroma DB.  
3. **Chat Interface** — Streamlit app connects the vector DB with a local Ollama LLM for intelligent Q&A.

---

## How to Use

1. Run the app:
   ```bash
   streamlit run chatbox_with_memory.py
2. Upload your documents from the interface sidebar.
3. Choose your preferred LLM model (DeepSeek, Mistral, Llama 3) and start chatting!

---

## Requirement

1. Python 3.10+
2. Ollama installed locally
3. Run this to install dependencies:
   ```bash
   pip install -r requirements.txt
