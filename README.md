# YouTube Corrective RAG System

An end-to-end, async Retrieval-Augmented Generation (RAG) system deployed on Google Cloud Run for question-answering over YouTube video content.

The application ingests a YouTube URL, processes and transcribes the audio via Groq's cloud API, embeds the transcript into a vector database, and serves an interactive Q&A API using a Corrective RAG workflow built with LangGraph.

## Live Demo & API Documentation

- **Interactive API Docs (Swagger UI):** https://yt-rag-service-207229110932.asia-south1.run.app/docs

*Note: The service is deployed on a scale-to-zero serverless tier. The first request may experience a cold-start delay while the container initializes.*

## Features

- **Cloud-based Transcription**: Downloads audio via `yt-dlp` and transcribes it using Groq's `whisper-large-v3-turbo` model.
- **Pre-Graph Ingestion Pipeline**: Handles video downloading, transcription, file cleanup, chunking, embedding, and vector indexing before the Q&A workflow.
- **Async FastAPI Backend**: Exposes HTTP endpoints for video ingestion and conversational Q&A.
- **Corrective RAG Workflow**: Uses LLM-based relevance evaluation, query rewriting, and fallback mechanisms.
- **LLM Relevance Evaluation**: Evaluates retrieved transcript chunks for relevance before generating the final answer.
- **Query Rewriting**: Rewrites the query when initial retrieval produces insufficient context, then retries retrieval once.
- **Web Search Fallback**: Falls back to DuckDuckGo if the rewritten query still produces inadequate context.
- **Centralized Logging**: Structured logging for local development and production monitoring.

## Architecture & Workflow

### 1. Pre-Graph Processing

Before the LangGraph workflow starts, the `POST /` endpoint:

1. Downloads audio from the YouTube URL using `yt-dlp`.
2. Transcribes the audio through Groq using `whisper-large-v3-turbo`.
3. Removes temporary audio files.
4. Chunks the transcript and generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores the embedded transcript chunks in ChromaDB.

### 2. Corrective RAG Workflow

When a query is sent to `POST /qna`:

1. **Retrieval**: Fetches the top-k transcript chunks from ChromaDB.
2. **Relevance Judgment**: An LLM-as-a-Judge evaluates the retrieved context.
3. **Query Rewriting**: If the context is insufficient, the Query Rewriter generates an improved query.
4. **Retrieval Retry**: Performs one additional retrieval using the rewritten query.
5. **Web Search Fallback**: If the retry still produces insufficient context, searches the web using DuckDuckGo.
6. **Response Generation**: Gemini generates the final answer using the available context.

See `YT RAG Flow Diagram` for the complete workflow.

## Performance Evaluation

Evaluated retrieval performance on a 10-query golden benchmark dataset:

| Metric | Result |
| :--- | ---: |
| Recall@5 | **100%** |
| Mean Reciprocal Rank (MRR) | **0.81** |
| Context Precision | **62.5%** |

*Note: The benchmark contains only 10 queries, so these results should be treated as an engineering evaluation rather than a statistically representative benchmark.*

## Tech Stack

- **Framework**: LangChain, LangGraph
- **API Engine**: FastAPI, Uvicorn
- **LLM**: Google Gemini 2.5 Flash Lite
- **Speech-to-Text**: Groq API (`whisper-large-v3-turbo`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: ChromaDB
- **Video/Audio**: `yt-dlp`, FFmpeg
- **Containerization & Deployment**: Docker, Google Cloud Run
- **Secrets & Registry**: GCP Secret Manager, GCP Artifact Registry

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check endpoint returning API status. |
| `POST` | `/` | Ingests a YouTube URL, transcribes the video, and initializes the vector store. |
| `POST` | `/qna` | Accepts a user query and returns a generated response using the active video context. |

### API Usage Example

**1. Ingest Video:**
```bash
curl -X POST "https://yt-rag-service-207229110932.asia-south1.run.app/" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
````

**2. Ask a Question:**

```bash
curl -X POST "https://yt-rag-service-207229110932.asia-south1.run.app/qna" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic of this video?"}'
```

## Local Setup & Development

### Prerequisites

* Python 3.11
* `ffmpeg` installed on system path
* Google Gemini API Key
* Groq API Key

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
ENVIRONMENT=local
```

### Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running Locally

```bash
uvicorn src.app:app --port 8080 --reload
```

Access local documentation at `http://localhost:8080/docs`.

## Deployment Architecture

The application is containerized using Docker and deployed on Google Cloud Run:

```text
[ Client ] ──HTTP──> [ Cloud Run / FastAPI ]
                           │
                           ├──> Groq API
                           ├──> ChromaDB
                           ├──> Gemini API
                           └──> GCP Secret Manager
```

* **Containerization**: Uses CPU-only PyTorch configuration to reduce image size.
* **Secrets Management**: API keys are injected at runtime via GCP Secret Manager.
* **Embedding Model**: Pre-downloaded during image build to avoid downloading during container startup.
* **Deployment**: Cloud Run with scale-to-zero enabled.

## Deployment & Production Limitations

* **Cold Starts**: Scale-to-zero can introduce startup latency after periods of inactivity.
* **Single Instance**: Configured with `max_instances=1` for portfolio cost management.
* **In-Memory Vector Store**: ChromaDB stores the active video context in memory. Data is lost when the container instance is terminated or restarted.
* **Single Video Context**: The current implementation supports one active video context per running instance and is not designed for concurrent multi-user workloads.
* **Web Search Rate Limits**: DuckDuckGo fallback may be rate-limited under higher request volumes.
* **Web Search Results**: DuckDuckGo API sometimes sometimes provides inaccurate websearch results. It is advisible to opt for a better WebSearch API for accuracy.
* **Evaluation Size**: Retrieval metrics are based on a 10-query golden dataset and don't include Websearch retrievals.

## License

MIT

## Author

Built as an AI/ML portfolio project demonstrating asynchronous GenAI application development, Corrective RAG workflows, LLM-based retrieval evaluation, and serverless cloud deployment.