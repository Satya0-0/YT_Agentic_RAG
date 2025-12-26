# YouTube Agentic RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system for question-answering over YouTube video content. 
This system downloads videos, transcribes them using Whisper, and enables query-based retrieval with adaptive fallback mechanisms like Query Re-writing, and Web Search.

## Features

- **Automated Video Download and Trascription**: Downloads YouTube video from provided URL to a specified local location and generates transcriptions using OpenAI's Whisper model (from HuggingFace). Upon creating the transcripts, the video is deleted
- **VectorDB Based Storaged and Retrieval**: Stores transcriptions in ChromaDB with semantic search capabilities
- **Agentic Workflow**: Multi-agent workflow demonstrating agentic decision-making
- **Query Rewriting**: Query reqriting using another LLM for improved retrieval accuracy
- **LLM-Based Relevance Judgment**: Evaluates retrieved documents' relevance before generating response
- **Web Search Fallback**: Fall back to web search for information retrieval (even after Query Rewriting)
- **Conversational Loop**: Supports multi-turn conversations with 'state' persistence

## Architecture

The system implements a graph-based workflow using LangGraph with the following stages:

1. **Video Download**: Downloads YouTube video audio using 'yt-dlp' to a local path
2. **Transcription**: Converts downloaded YouTube Video's audio to text using Whisper (openai/whisper-small)
3. **Video Clean-up**: Deletes the downloaded video from the local directory for better memory managment
4. **Vector Storage**: Chunks and embeds video transcription in ChromaDB 
5. **Retrieval**: Retrieves relevant document chunks based on user query (via sematic search)
6. **Relevance Judgment**: A separate LLM (LLM_J) evaluates if retrieved documents answer the query
7. **Query Rewriting**: Rewrite queries using another LLM (LLM_RQ) when they fail initial retrieval and the documents are retrieved again
8. **Web Search**: Fall back to DuckDuckGo search for unanswered queries
9. **Response Generation**: Generates final answer from context (either from VectorDB Docs or Web Search Results)

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash Lite
- **Speech-to-Text**: OpenAI Whisper (small and Local model)
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Framework**: LangChain + LangGraph
- **Video Download**: yt-dlp
- **Web Search**: DuckDuckGo API

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini)

## Installation

```bash
pip install yt-dlp transformers torch langchain langchain-google-genai
pip install langchain-community langchain-chroma sentence-transformers
pip install -qU duckduckgo-search langchain-community
pip install -U ddgs
pip install pydantic
```

## Configuration

Set-up your API keys as environment variables (refer ".env .example" for reference):

```python
GOOGLE_API_KEY=your_api_key_here
```

Update the local download path, and LLM models in the "config.py":

*By default, the videos are downloaded in a separate directory (`video_downloads`) within the Project's root folder on your computer. This is OS agnostic and can be overriden if required.*

## Usage

Execute the "main.py" script and provide the following when prompted:

1. YouTube URL
2. Initial question about the video content
3. Follow-up questions (type 'quit' to exit)

Example:

```
>> Enter the YouTube URL for Q&A: https://www.youtube.com/watch?v=tKPSmn-urB4
>> Enter your query: What is RAG?
```

The system will:
- Download, transcribe, and delete the video (only for the first run)
- Retrieve relevant chucks from it
- Generate a contextual answer (either from video or from Internet)
- Allow follow-up questions on the same video

## Workflow Logic

- **First Query**: Transcribes video, creates vector database, retrieves and answers
- **Subsequent Queries**: Reuses existing vector database for faster responses
- **Failed Retrieval**: Automatically rewrites query and retries
- **Persistent Failure**: Falls back to web search with rewritten query

## Key Components

### State Management
Uses Pydantic-based state tracking with fields for user query, retrieval flags, documents, and web results.

### Custom Retriever
Implements LangChain's BaseRetriever interface with ChromaDB similarity search (top-k=3).

### LLM Judge
Binary relevance classifier which determines if retrieved documents contain relevant information.

### Query Rewriter
Expands and contextualizes queries to improve retrieval on second attempt.

## Current Limitations

- Implemented locally in Python Scripts (not deployed as service yet)
- Single video per session (new URL requires restart)
- Web search from DuckDuckGo sometimes results in unusable data given this is a free tier version. The API can be changed for better results
- CPU-based Whisper inference (slow for long videos)

## License
MIT

## Author

Built as part of AI/ML portfolio demonstrating GenAI, RAG systems, and agentic workflows.
