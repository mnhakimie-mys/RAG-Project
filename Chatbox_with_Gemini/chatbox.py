import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# --- LANGCHAIN TOOLS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_classic.memory import ConversationBufferMemory

# --- 1. ENVIRONMENT CHECKS ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Stop the app early if keys are missing
if not all([GOOGLE_API_KEY, QDRANT_CLOUD_URL, QDRANT_API_KEY]):
    st.error("❌ Missing API Keys! Please check your .env file.")
    st.stop()

GEMINI_MODEL = "gemini-2.5-flash"

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="🚀 SmartChat Gemini", layout="wide")

# --- CORE FUNCTIONS ---

def load_and_process_files(uploaded_files):
    all_text_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        ext = os.path.splitext(file.name)[1].lower()
        if ext == ".pdf": loader = PyPDFLoader(tmp_path)
        elif ext == ".txt": loader = TextLoader(tmp_path)
        elif ext == ".docx": loader = Docx2txtLoader(tmp_path)
        else: continue
        docs = loader.load()
        for i, d in enumerate(docs):
            d.metadata["source"] = file.name
            d.metadata["page"] = i + 1
        all_text_docs.extend(docs)
        os.remove(tmp_path)
    return all_text_docs

def generate_expanded_queries(query, llm):
    """Agentic Query Expansion with output cleaning."""
    prompt = (
        f"You are a search expert. User asked: '{query}'. "
        "Generate exactly 2 alternative search queries using technical synonyms. "
        "Output ONLY the queries, one per line. No bullets or numbering."
    )
    response = llm.invoke(prompt)
    raw_lines = response.content.strip().split("\n")
    clean_queries = [line.strip().lstrip("-*•0123456789. ").strip() for line in raw_lines if line.strip()]
    return [query] + clean_queries[:2]

def build_advanced_knowledge_base(docs):
    """Parent-Child Strategy: Search Small (200), Read Large (1000)."""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    parent_docs = parent_splitter.split_documents(docs)
    st.session_state.parent_chunks = [p.page_content for p in parent_docs]

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    child_texts, child_metadatas = [], []
    
    for i, p_doc in enumerate(parent_docs):
        sub_snippets = child_splitter.split_text(p_doc.page_content)
        for snippet in sub_snippets:
            child_texts.append(snippet)
            child_metadatas.append({**p_doc.metadata, "parent_index": i})

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    db = QdrantVectorStore.from_texts(
        texts=child_texts,
        metadatas=child_metadatas,
        embedding=embeddings,
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
        collection_name="gemini_docs",
        force_recreate=True,
        prefer_grpc=True
    )
    return db

def clear_chat_history():
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# --- APP LAYOUT ---
with st.sidebar:
    st.title("⚙️ Settings")
    user_temp = st.slider("Temperature", 0.0, 1.0, 0.3)
    user_max_tokens = st.slider("Max Tokens", 100, 2000, 500)
    st.button("🧹 Clear Chat", on_click=clear_chat_history)

st.title("🚀 SmartChat Gemini")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="history", return_messages=True)

uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# --- DB REBUILD LOGIC (Fix: Detect file changes) ---
if uploaded_files:
    # Create a unique key based on file names and sizes
    file_fingerprint = "-".join([f"{f.name}_{f.size}" for f in uploaded_files])
    
    if "current_files" not in st.session_state or st.session_state.current_files != file_fingerprint:
        with st.spinner("Rebuilding Knowledge Base for new files..."):
            docs = load_and_process_files(uploaded_files)
            st.session_state.db = build_advanced_knowledge_base(docs)
            st.session_state.current_files = file_fingerprint
        st.success("Ready!")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, 
        google_api_key=GOOGLE_API_KEY, 
        temperature=user_temp, 
        max_output_tokens=user_max_tokens
    )

    # Display History
    history = st.session_state.chat_memory.load_memory_variables({})["history"]
    for m in history:
        with st.chat_message("user" if m.type == "human" else "assistant"):
            st.write(m.content)

    user_query = st.chat_input("Ask about your documents...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.spinner("Searching..."):
            expanded = generate_expanded_queries(user_query, llm)
            
            all_children = []
            for q in expanded:
                all_children.extend(st.session_state.db.similarity_search(q, k=3))
            
            # --- CONTEXT & METADATA TRACKING ---
            parent_ids = []
            sources_info = [] # NEW: To store unique source details
            
            for c in all_children:
                idx = c.metadata.get("parent_index")
                if idx is not None and idx not in parent_ids:
                    parent_ids.append(idx)
                    # Store source and page for the references button
                    sources_info.append({
                        "source": c.metadata.get("source", "Unknown"),
                        "page": c.metadata.get("page", "N/A")
                    })
            
            top_parents = parent_ids[:6]
            # Filter sources to match only the top 6 parents
            final_sources = sources_info[:len(top_parents)]

            if not top_parents:
                answer = "I don't have enough info in the documents."
            else:
                context_text = "\n\n".join([st.session_state.parent_chunks[i] for i in top_parents])

                prompt = f"""Answer the question ONLY using the provided CONTEXT.
                If the answer is not in the context, reply exactly:
                "I don't have enough info in the documents."

                CONTEXT:
                {context_text}

                QUESTION:
                {user_query}

                ANSWER:
                """
                response = llm.invoke(prompt)
                answer = response.content
            
            st.session_state.chat_memory.save_context({"human": user_query}, {"ai": answer})

        # --- DISPLAY ANSWER AND SOURCE BUTTON ---
        with st.chat_message("assistant"):
            st.write(answer)
            
            # THE SOURCE REFERENCE BUTTON
            if top_parents:
                with st.expander("📍 View Source References"):
                    for i, info in enumerate(final_sources):
                        st.markdown(f"**Source {i+1}:** {info['source']} — *Page {info['page']}*")