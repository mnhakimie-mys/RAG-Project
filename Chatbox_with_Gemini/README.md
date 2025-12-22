AI chatbot that allows users to upload PDFs, TXT, or DOCX files and ask questions grounded strictly in the uploaded content.  It uses **Gemini 2.5 Flash**, **Qdrant Vector Database**, and a **parent–child retrieval strategy** to deliver accurate, low-hallucination answers.

---

## ✨ Key Features

* **Parent-Child Retrieval**: Decouples search precision from context delivery by searching small snippets (200 chars) while providing full paragraphs (1000 chars) to the AI.
* **Agentic Query Expansion**: Uses an LLM "expert" to rewrite user queries into multiple technical synonyms, ensuring the system finds relevant data even if keywords don't match exactly.
* **Source Attribution**: Includes a **"View Source References"** expander that identifies the exact filename and page number for every piece of information used in the answer.
* **State-Aware UI**: 
    * **Auto-Rebuild**: Detects file changes (uploads/removals) and automatically re-indexes the database.
    * **Smart Clear**: Automatically wipes chat history when the document set changes to prevent cross-document confusion.
* **Hallucination Guardrails**: Implements strict system prompting and "Empty Context" safety checks to ensure the AI says *"I don't have enough info"* rather than guessing.

---

## 🧠 The Strategy for Accuracy

This project implements two core strategies to solve the "Precision vs. Context" trade-off common in AI applications:

### 1. The Parent-Child Bridge
* **Child Chunks (200 chars)**: Vectorized and stored in **Qdrant Cloud**. These provide the "search precision."
* **Parent Chunks (1000 chars)**: Linked via metadata and stored in session memory. These provide the "contextual depth."
* **The Result**: The AI finds the exact needle in the haystack but reads the whole page to understand the meaning.

### 2. Multi-Query Expansion
To bridge the gap between human language and technical documentation:
* The system generates **2 alternative queries** for every user question.
* It performs a **broadcast search** across all 3 queries.
* It filters and caps the results to the **Top 6 unique parent paragraphs** to keep the response focused and cost-efficient.

---

## 🛠️ Tech Stack & Models

| Component | Technology |
| :--- | :--- |
| LLM | Gemini 2.5 Flash |
| Embeddings | text-embedding-004 |
| Vector Database | Qdrant Cloud |
| Orchestration | LangChain |
| UI Framework | Streamlit |

---

## ⚙️ Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Environment**: Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_key_here
    QDRANT_CLOUD_URL=your_url_here
    QDRANT_API_KEY=your_key_here
    ```
3.  **Run the App**:
    ```bash
    streamlit run chatbox.py
    ```
