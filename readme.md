# 🎥 YouTube AI Q&A Bot (RAG-Based LLM Application)

An advanced **YouTube Question Answering System** powered by **Retrieval-Augmented Generation (RAG)** that enables users to ask questions about any YouTube video and receive **timestamp-aware, context-grounded answers**.

---

## 🚀 Features

- 🔍 **YouTube Transcript Ingestion**
  Extracts captions using `youtube_transcript_api` and converts them into timestamped chunks

- 🧠 **Advanced RAG Pipeline**
  Multi-stage retrieval with:
  - Semantic Search (HuggingFace `e5-base-v2`)
  - Vector Storage using Qdrant
  - BM25 Keyword Retrieval
  - Cross-Encoder Reranking

- 🔄 **Query Optimization**
  - Query rewriting
  - Multi-query expansion for better recall

- 🎯 **Accurate & Grounded Answers**
  - Timestamp-aware responses
  - JSON structured output (one sentence per timestamp)
  - Reduced hallucinations

- 🤖 **LLM Integration**
  - Local inference using ChatOllama (`phi3`)
  - Optional support for Google Gemini

- 🌐 **Chrome Extension UI**
  - Chat interface on YouTube
  - Auto-detects video ID
  - Jump-to-timestamp buttons
  - Per-video chat history

- ⚙️ **Modular Architecture**
  - ingestion_service
  - transcript_service
  - vector_service
  - RAG_service
  - qa_service
  - query_service
  - reranker_service
  - BM25_service

- 📊 **Token Usage Tracking**
  - Tracks input/output tokens for LLM calls

---

## 🏗️ Architecture

```text
YouTube Video
     ↓
Transcript Extraction
     ↓
Chunking + Embeddings
     ↓
Qdrant Vector DB
     ↓
Query → Rewrite → Expand
     ↓
Hybrid Retrieval (BM25 + Vector)
     ↓
Cross-Encoder Reranking
     ↓
LLM (Phi3 / Ollama)
     ↓
Timestamped Answer (JSON)
```

---

## 🧑‍💻 Tech Stack

- **Backend:** FastAPI, Python
- **LLM:** Ollama (Phi3), Gemini (optional)
- **Embeddings:** HuggingFace (intfloat/e5-base-v2)
- **Vector DB:** Qdrant
- **Search:** BM25 + Semantic Search
- **Frontend:** Chrome Extension
- **Others:** Docker, REST APIs

---

## 📌 API Endpoints

### 1️⃣ Process Video

```bash
GET /process?video_id=VIDEO_ID
```

- Extracts transcript
- Generates embeddings
- Stores in Qdrant

---

### 2️⃣ Ask Questions

```bash
GET /ask?query=YOUR_QUESTION&video_id=VIDEO_ID
```

- Returns timestamped, contextual answers

---

## 🧪 Example Response

```json
[
  {
    "timestamp": "00:01:23",
    "answer": "The speaker explains how embeddings improve semantic search."
  },
  {
    "timestamp": "00:02:10",
    "answer": "RAG pipelines combine retrieval with generation for better accuracy."
  }
]
```

---

## ⚡ Key Highlights

- 🔥 Hybrid Retrieval (BM25 + Embeddings)
- 🔥 Cross-Encoder Reranking for precision
- 🔥 Query Expansion for improved recall
- 🔥 Timestamp-grounded responses
- 🔥 Production-style modular architecture

---

## 📈 Use Cases

- 🎓 Learn from long YouTube lectures quickly
- 📊 Extract insights from podcasts
- 💼 Analyze technical talks
- 🧠 Build AI-powered video assistants

---

## 🛠️ Setup Instructions

```bash
# Clone repo
git clone https://github.com/shreyas-kapse/youtube_bot.git

cd youtube-ai-bot

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```

---

## 🔮 Future Improvements

- Real-time streaming QA
- Multi-video knowledge aggregation
- UI improvements with React
- Fine-tuned domain-specific LLM

---

## 🤝 Contributing

Feel free to fork this repo, open issues, or submit PRs 🚀

---

## ⭐ If you like this project

Give it a star ⭐ and connect with me on LinkedIn!
