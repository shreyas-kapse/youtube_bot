# 🎥 YouTube RAG Chatbot — AI-Powered Question Answering System for YouTube Videos

An advanced **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about YouTube videos and receive **context-aware, timestamp-grounded answers** using **LLMs, semantic search, hybrid retrieval, and reranking**.

Built using:

* FastAPI
* Ollama (Phi3)
* Qdrant Vector Database
* LangChain
* Hybrid Search (BM25 + Vector Search)
* Cross-Encoder Reranking
* LangSmith Observability
* RAGAS Evaluation Framework

---

# 🚀 Features

## 🎥 YouTube Transcript Processing

* Extracts captions using `youtube_transcript_api`
* Converts transcripts into timestamp-aware semantic chunks
* Stores embeddings in Qdrant vector database

---

## 🧠 Advanced RAG Pipeline

### Hybrid Retrieval System

Combines:

* Dense Vector Search (`intfloat/e5-base-v2`)
* BM25 Keyword Retrieval
* Multi-query Expansion
* Query Rewriting

for higher retrieval accuracy and recall.

---

## 🎯 Cross-Encoder Reranking

Uses reranking to:

* improve retrieval precision
* reduce noisy context
* improve answer grounding
* reduce hallucinations

---

## 🤖 Local LLM Inference

Supports:

* Ollama (`phi3`)
* Google Gemini (optional)

Features:

* local inference
* privacy-friendly setup
* low-cost deployment

---

## ⏱️ Timestamp-Aware Answers

Every answer is grounded to transcript timestamps for:

* explainability
* source attribution
* easy video navigation

---

## 🌐 Chrome Extension Interface

Interactive YouTube assistant with:

* in-video chat interface
* automatic video detection
* jump-to-timestamp functionality
* per-video chat history

---

## 📊 Observability & Evaluation

### LangSmith Tracing

Integrated LangSmith tracing for:

* prompt debugging
* retrieval inspection
* latency monitoring
* token tracking
* pipeline observability

---

### RAGAS Evaluation

Evaluated using RAGAS metrics:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

#### Current Scores

| Metric           | Score |
| ---------------- | ----- |
| Faithfulness     | 0.75  |
| Answer Relevancy | 0.71  |

---

# 🏗️ System Architecture

```text
YouTube Video
       ↓
Transcript Extraction
       ↓
Chunking + Embeddings
       ↓
Qdrant Vector Database
       ↓
Query Rewriting + Expansion
       ↓
Hybrid Retrieval
(BM25 + Vector Search)
       ↓
Cross-Encoder Reranking
       ↓
LLM (Phi3 / Gemini)
       ↓
Timestamp-Grounded Answers
```

---

# ⚡ Key Technical Highlights

* Hybrid Retrieval Architecture
* Semantic Search with Embeddings
* Cross-Encoder Reranking
* Query Expansion & Rewriting
* Timestamp-Grounded Responses
* Local LLM Integration
* LangSmith Observability
* RAGAS Benchmark Evaluation
* Production-Style Modular Design

---

# 🧩 Modular Architecture

```text
services/
├── ingestion_service
├── transcript_service
├── vector_service
├── rag_service
├── qa_service
├── reranker_service
├── bm25_service
├── query_service
```

---

# 🛠️ Tech Stack

| Category        | Technologies                      |
| --------------- | --------------------------------- |
| Backend         | FastAPI, Python                   |
| LLMs            | Ollama (Phi3), Gemini             |
| Embeddings      | HuggingFace `intfloat/e5-base-v2` |
| Vector Database | Qdrant                            |
| Retrieval       | BM25 + Semantic Search            |
| Reranking       | Cross-Encoder                     |
| Frontend        | Chrome Extension                  |
| Evaluation      | RAGAS                             |
| Observability   | LangSmith                         |

---

# 📌 API Endpoints

## 1️⃣ Process YouTube Video

```bash
GET /process?video_id=VIDEO_ID
```

### Functionality

* extracts transcript
* generates embeddings
* stores vectors in Qdrant

---

## 2️⃣ Ask Questions

```bash
GET /ask?query=YOUR_QUESTION&video_id=VIDEO_ID
```

### Returns

* contextual answers
* timestamp-aware responses
* grounded retrieval-based outputs

---

# 🧪 Example Response

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

# 📈 RAG Optimization Techniques Used

## Retrieval Improvements

* Hybrid Search
* Query Expansion
* Query Rewriting
* Video-Specific Filtering

---

## Generation Improvements

* Context Grounding
* Hallucination Reduction
* Prompt Optimization

---

## Retrieval Quality Improvements

* Cross-Encoder Reranking
* Deduplication
* Top-K Optimization

---

# 📊 Evaluation Pipeline

The chatbot was evaluated using:

* custom DSA-focused evaluation dataset
* RAGAS evaluation framework
* LangSmith tracing

Evaluation workflow:

```text
Question
    ↓
Retriever
    ↓
Retrieved Context
    ↓
LLM Generation
    ↓
RAGAS Evaluation
```

---

# 💼 Use Cases

* Learn from long YouTube lectures efficiently
* Build AI video assistants
* Analyze technical talks and tutorials
* Extract knowledge from podcasts
* Educational AI applications
* Context-aware video search systems

---

# 🛠️ Setup Instructions

## 1️⃣ Clone Repository

```bash
git clone https://github.com/shreyas-kapse/youtube_bot.git
```

```bash
cd youtube_bot
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Start Ollama

```bash
ollama serve
```

---

## 4️⃣ Pull Required Models

```bash
ollama pull phi3
```

```bash
ollama pull nomic-embed-text
```

---

## 5️⃣ Run FastAPI Server

```bash
uvicorn main:app --reload
```

---

# 🔮 Future Improvements

* Streaming token responses
* Multi-video knowledge aggregation
* Conversational memory
* Agentic RAG workflows
* Self-RAG / CRAG
* Graph-based workflows with LangGraph
* React frontend dashboard

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit pull requests.

---

# ⭐ Support

If you found this project useful:

* Star the repository ⭐
* Connect with me on LinkedIn
* Share feedback and suggestions

---