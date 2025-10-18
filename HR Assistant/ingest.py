import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "docs"
DATA_PATH = "data"

ch_size = 600
ch_overlap = 80

def load_documents():
    docs = []

    for filename in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)           
            pdf_pages = loader.load()
            
            for i, page in enumerate(pdf_pages, start=1):
                page.metadata["source"] = filename
                page.metadata["page"] = i
                docs.append(page)

    return docs

def split_chunk():
    docs = load_documents()
    print(f"[✅] Loaded {len(docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=ch_size, chunk_overlap=ch_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[✅] Split into {len(chunks)} chunks.")

    return chunks

def save():
    chunks = split_chunk()

    output_file = os.path.join(DATA_PATH, "chunks.txt")
    
    f = open(output_file, "w", encoding="utf-8")
    for i, chunk in enumerate(chunks, start=1):
        f.write(F"Chunk {i}\n{chunk.page_content}\n\n")
    f.close()

    print(f"[✅] Saved all chunks to: {output_file}")

if __name__ == "__main__":
    save()
