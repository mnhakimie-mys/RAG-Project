import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# --- LANGCHAIN TOOLS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_classic.memory import ConversationBufferMemory

# =============================================================================
# App Configuration
# =============================================================================

APP_TITLE = "🚀 SmartChat Gemini"
APP_PAGE_TITLE = "🚀 SmartChat Gemini"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
QDRANT_COLLECTION = "gemini_docs"

# Parent-Child (Search Small, Read Large)
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 200
CHILD_CHUNK_OVERLAP = 20

# Retrieval configuration
QUERY_EXPANSIONS = 2          # how many alternative queries to generate
CHILD_HITS_PER_QUERY = 3      # similarity_search k for each expanded query
MAX_PARENT_CONTEXT = 6        # max number of parent chunks injected into context


@dataclass(frozen=True)
class SourceRef:
    source: str
    page: int


# =============================================================================
# Environment & Secrets
# =============================================================================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

missing = [k for k, v in {
    "GOOGLE_API_KEY": GOOGLE_API_KEY,
    "QDRANT_CLOUD_URL": QDRANT_CLOUD_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
}.items() if not v]

if missing:
    st.error(f"❌ Missing required environment variables: {', '.join(missing)}. Please check your .env file.")
    st.stop()


# =============================================================================
# Streamlit UI Setup
# =============================================================================

st.set_page_config(page_title=APP_PAGE_TITLE, layout="wide")


def init_session_state() -> None:
    """Initialize all session state keys used by the app."""
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    if "db" not in st.session_state:
        st.session_state.db = None

    if "parent_chunks" not in st.session_state:
        st.session_state.parent_chunks = []

    if "current_files_fingerprint" not in st.session_state:
        st.session_state.current_files_fingerprint = None


def clear_chat_history() -> None:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="history", return_messages=True)


# =============================================================================
# File Loading & Knowledge Base
# =============================================================================

def _write_to_tempfile(uploaded_file) -> str:
    """Write uploaded Streamlit file to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def load_documents(uploaded_files) -> List:
    """
    Load uploaded files into LangChain Document objects.
    Supports: PDF, TXT, DOCX.
    Adds metadata: source (filename), page (1-based page index).
    """
    docs = []

    for file in uploaded_files:
        tmp_path = _write_to_tempfile(file)
        ext = os.path.splitext(file.name)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".txt":
                loader = TextLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                continue

            loaded = loader.load()

            # Ensure consistent metadata for downstream UI + citations
            for i, d in enumerate(loaded, start=1):
                d.metadata["source"] = file.name
                d.metadata["page"] = i

            docs.extend(loaded)

        finally:
            # Always remove temp file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return docs


def build_vector_store(docs: List) -> QdrantVectorStore:
    """
    Build a Parent-Child retrieval index:
    - Parent chunks: larger context blocks
    - Child chunks: smaller searchable snippets, each points to a parent_index
    """
    # Split into parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    parent_docs = parent_splitter.split_documents(docs)

    # Store parent text for later “read large”
    st.session_state.parent_chunks = [p.page_content for p in parent_docs]

    # Split parents into child snippets (search small)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    child_texts = []
    child_metadatas = []

    for parent_index, p_doc in enumerate(parent_docs):
        snippets = child_splitter.split_text(p_doc.page_content)
        for snippet in snippets:
            child_texts.append(snippet)
            child_metadatas.append({**p_doc.metadata, "parent_index": parent_index})

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # NOTE: force_recreate=True will rebuild the whole collection each time
    # new files are detected. This keeps things simple and consistent.
    return QdrantVectorStore.from_texts(
        texts=child_texts,
        metadatas=child_metadatas,
        embedding=embeddings,
        url=QDRANT_CLOUD_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
        force_recreate=True,
        prefer_grpc=True,
    )


def get_files_fingerprint(uploaded_files) -> str:
    """Create a stable fingerprint for detecting file set changes."""
    return "|".join([f"{f.name}:{f.size}" for f in uploaded_files])


# =============================================================================
# Query Expansion & Retrieval
# =============================================================================

def generate_expanded_queries(user_query: str, llm: ChatGoogleGenerativeAI) -> List[str]:
    """
    Generate alternative queries to improve recall.
    Returns: [original_query, alt1, alt2]
    """
    prompt = (
        "You are a search expert.\n"
        f"User asked: {user_query!r}\n\n"
        f"Generate exactly {QUERY_EXPANSIONS} alternative search queries using technical synonyms.\n"
        "Rules:\n"
        "- Output ONLY the queries\n"
        "- One per line\n"
        "- No bullets, no numbering\n"
    )

    resp = llm.invoke(prompt)
    lines = [ln.strip() for ln in resp.content.splitlines() if ln.strip()]

    # Defensive cleanup: remove common bullet/number prefixes
    cleaned = []
    for ln in lines:
        cleaned.append(ln.lstrip("-*•0123456789. ").strip())

    # Guarantee: original + up to QUERY_EXPANSIONS alts
    alts = cleaned[:QUERY_EXPANSIONS]
    return [user_query] + alts


def retrieve_parent_context(
    expanded_queries: List[str],
    db: QdrantVectorStore,
) -> Tuple[List[int], List[SourceRef]]:
    """
    1) Search child chunks for each expanded query
    2) Collect unique parent_index values (preserve order)
    3) Return top parent ids and matching source references
    """
    parent_ids: List[int] = []
    source_refs: List[SourceRef] = []

    for q in expanded_queries:
        hits = db.similarity_search(q, k=CHILD_HITS_PER_QUERY)
        for doc in hits:
            idx = doc.metadata.get("parent_index")
            if idx is None or idx in parent_ids:
                continue

            parent_ids.append(idx)

            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)
            try:
                page_int = int(page)
            except (TypeError, ValueError):
                page_int = 0

            source_refs.append(SourceRef(source=src, page=page_int))

            if len(parent_ids) >= MAX_PARENT_CONTEXT:
                return parent_ids, source_refs

    return parent_ids, source_refs


def answer_from_context(user_query: str, context_text: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Constrained answering: only use provided context.
    If not found: return the exact fallback string.
    """
    prompt = f"""
Answer the question ONLY using the provided CONTEXT.

If the answer is not in the context, reply exactly:
I don't have enough info in the documents.

CONTEXT:
{context_text}

QUESTION:
{user_query}

ANSWER:
""".strip()

    resp = llm.invoke(prompt)
    return resp.content.strip()


# =============================================================================
# UI
# =============================================================================

init_session_state()

with st.sidebar:
    st.title("⚙️ Settings")
    user_temp = st.slider("Temperature", 0.0, 1.0, 0.3, help="Higher = more creative, lower = more precise.")
    user_max_tokens = st.slider("Max Tokens", 100, 2000, 500, help="Controls maximum output length.")
    st.button("🧹 Clear Chat", on_click=clear_chat_history)

st.title(APP_TITLE)

uploaded_files = st.file_uploader(
    "Upload files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload documents to build a knowledge base, then ask questions in the chat.")
    st.stop()

# Build / rebuild DB when file set changes
new_fingerprint = get_files_fingerprint(uploaded_files)
if st.session_state.current_files_fingerprint != new_fingerprint:
    with st.spinner("Building knowledge base..."):
        docs = load_documents(uploaded_files)
        st.session_state.db = build_vector_store(docs)
        st.session_state.current_files_fingerprint = new_fingerprint
    st.success("✅ Knowledge base is ready.")

# LLM instance (uses sidebar settings)
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=user_temp,
    max_output_tokens=user_max_tokens,
)

# Render chat history
history = st.session_state.chat_memory.load_memory_variables({}).get("history", [])
for m in history:
    role = "user" if m.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(m.content)

user_query = st.chat_input("Ask about your documents...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Searching and answering..."):
        expanded_queries = generate_expanded_queries(user_query, llm)

        parent_ids, source_refs = retrieve_parent_context(
            expanded_queries=expanded_queries,
            db=st.session_state.db,
        )

        if not parent_ids:
            answer = "I don't have enough info in the documents."
            source_refs = []
        else:
            context_text = "\n\n".join(st.session_state.parent_chunks[i] for i in parent_ids)
            answer = answer_from_context(user_query, context_text, llm)

        st.session_state.chat_memory.save_context({"human": user_query}, {"ai": answer})

    with st.chat_message("assistant"):
        st.write(answer)

        if source_refs:
            with st.expander("📍 Source References"):
                for i, ref in enumerate(source_refs, start=1):
                    page_label = f"Page {ref.page}" if ref.page > 0 else "Page N/A"
                    st.markdown(f"**Source {i}:** {ref.source} — *{page_label}*")
