# YouTube RAG Bot — AI-Powered YouTube Video Q&A using RAG, Hybrid Search & LLMs

> Ask any question about a YouTube video and get **timestamp-grounded answers** in seconds — powered by Retrieval-Augmented Generation (RAG), hybrid BM25 + semantic search, cross-encoder reranking, and local LLMs via Ollama.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-blue)](https://langchain.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)](https://qdrant.tech)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)](https://ollama.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-purple)](https://docs.ragas.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-orange)](https://smith.langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What Is YouTube RAG Bot?

**YouTube RAG Bot** is a production-grade RAG system that turns any YouTube video into an interactive knowledge base. Point it at a video, and you can ask natural language questions and get precise, timestamped answers grounded in what was actually said — not hallucinated.

It ships with a **Chrome Extension** that embeds directly into the YouTube UI so you can chat with a video while watching it, with clickable timestamps that jump you to the exact moment in the video.

---

## Features

- **Hybrid retrieval** — combines dense vector search (`intfloat/e5-base-v2`) with BM25 keyword search, then merges and deduplicates results for higher recall than either approach alone
- **Query rewriting + expansion** — LLM rewrites and expands the user's query into multiple variants before retrieval, catching relevant content the original phrasing would miss
- **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores all retrieved chunks for precision, reducing noisy context passed to the LLM
- **Timestamp-grounded answers** — every sentence in the response is tied to a `MM:SS` timestamp from the original video, making answers verifiable and the video navigable
- **Chrome Extension** — chat with any YouTube video directly from the browser; automatic video ID detection, per-video chat history, clickable jump-to-timestamp buttons
- **Local LLM inference** — runs fully on-device with Ollama (Phi3); no API keys needed, no data sent to the cloud, zero cost per query
- **Multi-LLM support** — switch to Google Gemini with one line in the config for higher answer quality
- **RAGAS evaluation** — pipeline benchmarked on Faithfulness (0.75) and Answer Relevancy (0.71) using a custom evaluation dataset
- **LangSmith observability** — full trace visibility for every retrieval and generation step; latency, token counts, and prompt inspection out of the box

---

## Architecture

```
YouTube Video URL
       │
       ▼
Transcript Extraction          ← youtube_transcript_api
       │
       ▼
Chunking (200-word windows)    ← timestamp-aware segments
       │
       ▼
Embedding (intfloat/e5-base-v2)
       │
       ▼
Qdrant Vector Store            ← per-video namespace filtering
       │
──── Query Time ────────────────────────────────────────
       │
       ▼
Query Rewriting + Expansion    ← LLM generates 3 variants
       │
       ▼
Hybrid Retrieval
├── Dense Vector Search        ← semantic similarity
└── BM25 Keyword Search        ← exact term matching
       │
       ▼
Deduplication + Merging
       │
       ▼
Cross-Encoder Reranking        ← ms-marco-MiniLM-L-6-v2
       │
       ▼
Top-5 Context Chunks
       │
       ▼
LLM Generation (Phi3 / Gemini)
       │
       ▼
Timestamp-Grounded JSON Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| RAG Framework | LangChain |
| LLM (local) | Ollama — Phi3 |
| LLM (cloud) | Google Gemini |
| Embeddings | HuggingFace `intfloat/e5-base-v2` |
| Vector Database | Qdrant |
| Keyword Search | BM25 (rank-bm25) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Transcript Extraction | youtube-transcript-api |
| Chrome Extension | Manifest V3 |
| Evaluation | RAGAS |
| Observability | LangSmith |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed
- [Qdrant](https://qdrant.tech) running locally (Docker recommended)

### 1. Clone and install

```bash
git clone https://github.com/shreyas-kapse/youtube_bot.git
cd youtube_bot
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Pull Ollama models

```bash
ollama pull phi3
ollama pull nomic-embed-text
```

### 4. Configure environment

Create a `.env` file:

```env
# Optional: switch to Gemini
GOOGLE_API_KEY=your_google_api_key

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=youtube-rag-bot
```

### 5. Start the server

```bash
uvicorn main:app --reload
```

---

## API Endpoints

### Process a YouTube Video

```bash
GET /process?video_id=VIDEO_ID
```

Extracts the transcript, generates embeddings, and stores vectors in Qdrant. Run this once per video before asking questions.

### Ask a Question

```bash
GET /ask?query=YOUR_QUESTION&video_id=VIDEO_ID
```

Runs the full RAG pipeline — query rewriting, hybrid retrieval, reranking, LLM generation — and returns a timestamped answer.

**Example response:**

```json
{
  "segments": [
    {
      "sentence": "The speaker explains how embeddings capture semantic meaning for better search.",
      "timestamp": "01:23"
    },
    {
      "sentence": "RAG pipelines combine retrieval with generation to reduce hallucinations.",
      "timestamp": "02:10"
    }
  ]
}
```

---

## Chrome Extension

The extension embeds a chat sidebar directly into YouTube.

**Setup:**

1. Open Chrome → `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `youtube-qa-extension/` folder
4. Open any YouTube video and click the extension icon

**Features:**
- Auto-detects the current video ID from the URL
- Chat history preserved per video
- Each answer sentence has a **Jump to timestamp** button that seeks the video to that moment

---

## RAG Pipeline — Technical Details

### Retrieval Strategy

Standard RAG sends one query to a vector store. YouTube RAG Bot does more:

1. **Query rewriting** — the LLM rewrites the question to be more precise for semantic search
2. **Query expansion** — generates 3 alternate phrasings, retrieves for all of them
3. **Hybrid merge** — BM25 results are weighted ×2 and merged with vector results (BM25 catches exact terminology that semantic search misses)
4. **Deduplication** — removes duplicate chunks before reranking
5. **Cross-encoder reranking** — all candidates are re-scored with a cross-encoder for final precision

This multi-stage pipeline consistently outperforms single-query vector search on technical content.

### Chunking

Transcripts are chunked into 200-word windows with `start_time` and `end_time` metadata preserved on every chunk. The LLM prompt instructs the model to attach exactly one timestamp per sentence in the response.

---

## RAGAS Evaluation

The pipeline was benchmarked using RAGAS on a custom evaluation dataset.

| Metric | Score |
|---|---|
| Faithfulness | 0.75 |
| Answer Relevancy | 0.71 |

**Faithfulness** measures whether the generated answer is grounded in the retrieved context (hallucination resistance). **Answer Relevancy** measures how directly the answer addresses the question.

To run the evaluation yourself:

```bash
python evaluation/run_ragas.py
```

Results are saved to `evaluation/ragas_results.json`.

---

## Project Structure

```
youtube_bot/
├── main.py                        # FastAPI app, lifespan, endpoints
├── service/
│   ├── ingestion_service.py       # Transcript → embeddings → Qdrant
│   ├── transcript_service.py      # YouTube transcript extraction + chunking
│   ├── vector_service.py          # Qdrant read/write, per-video filtering
│   ├── embedding_service.py       # HuggingFace e5-base-v2 singleton
│   ├── RAG_service.py             # Full RAG pipeline orchestration
│   ├── query_service.py           # Query rewriting + expansion
│   ├── BM25_service.py            # Keyword retrieval
│   ├── reranker_service.py        # Cross-encoder reranking
│   ├── LLM_service.py             # Multi-LLM factory (Ollama / Gemini)
│   └── qa_service.py              # Public API — process + answer
├── evaluation/
│   ├── eval_dataset.json          # Custom Q&A evaluation set
│   ├── generate_eval_answers.py   # Generate answers for evaluation
│   ├── run_ragas.py               # RAGAS evaluation runner
│   └── ragas_results.json         # Benchmark results
├── youtube-qa-extension/
│   ├── manifest.json              # Chrome Extension Manifest V3
│   ├── popup.html                 # Chat UI
│   └── popup.js                   # Extension logic
└── utils/
    └── logger.py
```

---

## Roadmap

- [ ] Docker Compose — one-command setup (app + Qdrant + Ollama)
- [ ] Streaming token responses
- [ ] Multi-video knowledge aggregation (ask across multiple videos)
- [ ] Conversational memory across turns
- [ ] Agentic RAG with LangGraph
- [ ] React frontend dashboard

---

## Related Projects

- [AI PR Reviewer](https://github.com/shreyas-kapse/ai_pr_reviewer) — Multi-agent LangGraph pipeline for automated GitHub pull request reviews

---

## License

[MIT](LICENSE)

---

*Built with LangChain, Qdrant, Ollama, FastAPI, and the YouTube Transcript API.*