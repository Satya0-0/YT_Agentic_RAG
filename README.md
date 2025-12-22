# YouTube Agentic RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system for question-answering over YouTube video content. 
This system downloads videos, transcribes them using Whisper, and enables query-based retrieval with adaptive fallback mechanisms like Query Re-writing, and Web Search.

## Features

- **Automated Video Download and Trascription**: Downloads YouTube video to the given location and generates transcriptions using OpenAI's Whisper model (from HuggingFace)
- **VectorDB Based Storaged and Retrieval**: Stores transcriptions in ChromaDB with semantic search capabilities
- **Agentic Workflow**: Multi-stage query processing demonstrating agentic decision-making
- **Query Rewriting**: Adaptive query optimization using another LLM for improved retrieval accuracy
- **LLM-Based Relevance Judgment**: Evaluates retrieved document relevance before response generation
- **Web Search Fallback**: Automatic web search capability when retrieval fails (even after Query Rewriting)
- **Conversational Interface**: Supports multi-turn conversations with 'state' persistence

## Architecture

The system implements a graph-based workflow using LangGraph with the following stages:

1. **Video Download**: Downloads YouTube video audio using 'yt-dlp'
2. **Transcription**: Converts downloaded YouTube Video's audio to text using Whisper (openai/whisper-small)
3. **Vector Storage**: Chunks and embeds transcription in ChromaDB
4. **Retrieval**: Fetches relevant document chunks based on user query
5. **Relevance Judgment**: LLM evaluates if retrieved documents answer the query
6. **Query Rewriting**: Optimizes queries they fail initial retrieval and the documents are retrieved again
7. **Web Search**: Fall back to DuckDuckGo search for unanswered queries
8. **Response Generation**: Generates final answer from context (either from VectorDB Docs or Web Search Results)

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash 
- **Speech-to-Text**: OpenAI Whisper (small model - Local model)
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Framework**: LangChain + LangGraph
- **Video Download**: yt-dlp
- **Web Search**: DuckDuckGo API

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini)
- Serper API Key (for web search)

## Installation

```bash
pip install yt-dlp transformers torch langchain langchain-google-genai
pip install langchain-community langchain-chroma sentence-transformers
pip install -qU duckduckgo-search langchain-community
pip install -U ddgs
pip install pydantic
```

## Configuration

Set-up your API keys as environment variables:

```python
os.environ['GOOGLE_API_KEY'] = 'your-google-api-key'
```

Update the local download path in the code:

```python
localpath: str = r"your/downloads/path"
```

## Usage

Run the Jupyter notebook and provide:

1. YouTube URL when prompted
2. Initial question about the video content
3. Follow-up questions (type 'quit' to exit)

Example:

```
>> Enter the YouTube URL for Q&A: https://www.youtube.com/watch?v=tKPSmn-urB4
>> What is RAG?
```

The system will:
- Download and transcribe the video (only for the first run)
- Retrieve relevant segments
- Generate a contextual answer
- Allow follow-up questions on the same video

## Workflow Logic

- **First Query**: Downloads video, creates vector database, retrieves and answers
- **Subsequent Queries**: Reuses existing vector database for faster responses
- **Failed Retrieval**: Automatically rewrites query and retries
- **Persistent Failure**: Falls back to web search with rewritten query

## Key Components

### State Management
Uses Pydantic-based state tracking with fields for query history, retrieval flags, documents, and web results.

### Custom Retriever
Implements LangChain's BaseRetriever interface with ChromaDB similarity search (top-k=3).

### LLM Judge
Binary relevance classifier that determines if retrieved documents contain answer-relevant information.

### Query Rewriter
Expands and contextualizes queries to improve retrieval on second attempt.

## Current Limitations

- Runs locally in Jupyter notebook (not deployed as service yet)
- Video downloads stored permanently (no cleanup)
- Single video per session (new URL requires restart)
- Web search from DuckDuckGo sometimes results in unusable data given this is a free tier version. The API can be changed to Serper for better results
- CPU-based Whisper inference (slow for long videos)

## Future Enhancements (to be made by 10-Jan-2025):
- Production deployment with Python Script
- Automatic video cleanup

## License
MIT

## Author

Built as part of AI/ML portfolio demonstrating GenAI, RAG systems, and agentic workflows.
