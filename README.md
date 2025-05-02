# 📚 RAG Textbook QA

A Retrieval-Augmented Generation (RAG) system that uses a textbook as a source to answer user queries through a Flask API.

---

## 🚀 Features

- PDF ingestion and chunking
- Embedding and vector storage
- Semantic retrieval using similarity search
- LLM-based answer generation
- Flask-based API with Pydantic input validation
- Exposable via ngrok for public access

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ktxdev/rag_textbook_qa.git
cd rag_textbook_qa
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the App
```bash
python main.py
```
If set up correctly, Flask will start at:
```cpp
http://127.0.0.1:5000
```
## Sample request
```bash
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What is NLP?"}'
```
