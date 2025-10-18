from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import split_chunk

DATA_PATH = "data/chroma_db"

chunks = split_chunk()

embedding_f = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_f, persist_directory=DATA_PATH)
vectordb.persist()

print(f"[✅] Vector database saved to: '{DATA_PATH}'")