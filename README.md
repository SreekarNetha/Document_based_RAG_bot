# Document_based_RAG_bot


A Streamlit-based application that allows users to upload documents (PDF/DOCX) and ask questions about their content using embeddings and a Large Language Model (LLM).

---

## 🚀 Features

* 📂 Upload PDF and DOCX files
* ✂️ Automatic text extraction and chunking
* 🧠 Semantic search using embeddings
* 💬 Ask questions about your document
* 🔗 LLM integration (Groq / OpenAI-compatible APIs)
* 🗂 Persistent vector storage with ChromaDB
* 📝 Interaction logging for debugging and analysis

---

## 🏗️ Tech Stack

* **Frontend/UI:** Streamlit
* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector DB:** ChromaDB
* **LLM API:** Groq / OpenAI-compatible endpoints
* **Document Parsing:** PyPDF2, python-docx
* **Text Splitting:** LangChain

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

Inside the app UI:

1. Upload a **PDF or DOCX** file
2. Enter your **LLM API endpoint** (default provided)
3. Provide your **API Key**
4. Optionally change the **model name**

---

## 🧠 How It Works

1. **Document Upload**

   * Extracts text from PDF/DOCX files

2. **Text Processing**

   * Splits text into chunks using `RecursiveCharacterTextSplitter`

3. **Embedding Generation**

   * Converts chunks into vector embeddings

4. **Storage**

   * Stores embeddings in ChromaDB

5. **Query Handling**

   * Converts user query into embedding
   * Retrieves top relevant chunks

6. **LLM Response**

   * Sends context + query to LLM
   * Displays generated answer

---

## 📝 Logging

All interactions are stored in:

```
app_interactions.jsonl
```

Each entry includes:

* Timestamp
* User query
* Model response
* Model used

---

## 🔐 Environment Variables (Optional)

Instead of entering API keys in the UI, you can set:

```bash
export API_KEY=your_api_key
```

(You’ll need to modify the code slightly to use this.)

---

## ⚠️ Notes

* Ensure your API key has access to the selected model
* Large documents may take time to process
* Only PDF and DOCX formats are supported

---

## 💡 Future Improvements

* Support more file formats (TXT, HTML, etc.)
* Add chat history UI
* Streaming responses
* Multi-document querying
* Deployment (Docker / Cloud)

---

## 📜 License

This project is open-source and available under the MIT License.

---

If you want, I can also **pin exact versions (recommended for deployment)** or create a **Dockerfile** for this project.
