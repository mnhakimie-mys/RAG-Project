import os
import tempfile
import streamlit as st
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Parameters
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
DB_PATH = "data/chroma_db"  # optional persistence

# Streamlit setup
st.set_page_config(page_title="Smart Chatbox", layout="wide")
st.title("Smart Chatbox")
st.write("Upload documents (PDF, TXT, DOCX) and start chat!")
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

with st.sidebar:
    st.header("⚙️ Settings")
    llm_model = st.selectbox(
        "Select LLM model:",
        ["deepseek-r1", "mistral", "llama3"],
        index=0,
        help="Select a local Ollama model. Make sure it's already pulled."
    )
    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.number_input("Max Tokens", value=512, step=64, min_value=128, max_value=4096)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

def load_documents(files):
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        ext = os.path.splitext(file.name)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".txt":
                loader = TextLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                st.warning(f"⚠️ Unsupported file type: {file.name}")
                continue

            docs = loader.load()
            for i, d in enumerate(docs, start=1):
                d.metadata["source"] = file.name
                d.metadata["page"] = i 
            all_docs.extend(docs)

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
        finally:
            # Clean up the temporary file
            os.remove(tmp_path)
            
    return all_docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    vectordb = Chroma.from_documents(chunks, embedding_fn, persist_directory=DB_PATH)
    
    try:
        vectordb.persist()
    except:
        pass

    st.success("✅ Documents processed. You may start asking questions about them!")
    return vectordb

if uploaded_files:
    if (
        'last_uploaded' not in st.session_state
        or st.session_state.last_uploaded != [f.name for f in uploaded_files]
    ):
        st.session_state.last_uploaded = [f.name for f in uploaded_files]
        st.session_state.vectordb = None
        st.session_state.memory.clear()

        with st.spinner("Processing documents... this may take a moment..."):
            docs = load_documents(uploaded_files)
            if docs:
                st.session_state.vectordb = build_vectorstore(docs)
    
if st.session_state.vectordb:
    
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=llm_model, temperature=temperature, max_tokens=max_tokens)

    prompt_template = """
    You are a helpful AI assistant that answers based on provided context and previous conversation memory.
    If the answer is in the document context, explain it clearly with short, easy-to-read points.
    If the question is outside the context, respond politely that you don't have that information by saying "I can only answer questions based on the documents you uploaded."
    
    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    if st.button("🧹 Clear Chat"):
        st.session_state.memory.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("Conversation")
    
    history_messages = st.session_state.memory.load_memory_variables({})["chat_history"]
    
    for message in history_messages:
        role = "user" if message.type == "human" else "assistant"
        st.chat_message(role).write(message.content)

    user_input = st.chat_input("💬 Ask a question...")
    if user_input:        
        st.chat_message("user").write(user_input)
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"question": user_input})            
                st.chat_message("assistant").write(result["answer"])
            except Exception as e:
                st.error(f"An error occurred during LLM invocation: {e}")

else:
    st.info("👆 Upload a document first (PDF, TXT, DOCX) to start chatting with your data!")