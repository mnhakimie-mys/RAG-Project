import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "data/chroma_db"

st.set_page_config(page_title="HR Assistant", layout="wide")
st.title("HR Assistant")
st.write("Ask any question related to HR")

embedding_f = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_f)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="deepseek-r1", temperature=0.2, max_tokens=512)

prompt = ChatPromptTemplate.from_template(
    "You are HR assistant. Use the following context to answer the question accurately and concisely.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

rag_chain = prompt | llm | StrOutputParser()

user_q = st.text_input("Your question:")
if user_q:
    with st.spinner("Searching for answer..."):
        docs = retriever.invoke(user_q)
        context = "\n\n".join([d.page_content for d in docs])
        answer = rag_chain.invoke({"context": context, "question": user_q})

        st.markdown("### Answer:")
        st.write(answer)

        st.markdown("### Sources:")
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:300].replace("\n", " ")
            st.caption(f"{src}: {preview}...")
