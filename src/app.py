from fastapi import FastAPI, Security, HTTPException, status, Depends
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
from src.rag import Transcribe, QnALoop, BuildGraph
import os
from dotenv import load_dotenv

# Defining Global Vector Store
VECTOR_STORE = None

# Loading the API_Keys for Gemini and Groq
load_dotenv()

# Checking if the Keys are loaded appropriately
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing API Keys for GOOGLE_API_KEY or GROQ_API_KEY in environment variables!")

app = FastAPI()

class YT_URL(BaseModel):
    url: str
    
    model_config={
        "json_schema_extra": {
            "examples": [{
                "url": "YouTube URL to be used for Q&A"
            }]
        }
    }
    
class RequestQuery(BaseModel):
    query: str
    model_config={
        "json_schema_extra": {
            "examples": [{
                "query": "User Query for Q&A"
            }]
        }
    }

class RAGResponse(BaseModel):
    response: str

# A default GET method to check Health Status of the app
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "message": "YouTube Agentic RAG API is running.",
        "docs_url": "/docs"
    }

# Actual ingestion point of YouTUbe URL required for Q&A
@app.post("/", status_code=status.HTTP_201_CREATED)
async def read_url(video_qa: YT_URL):
    global VECTOR_STORE
    try:
        VECTOR_STORE = await Transcribe(video_qa.url)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail = "Issue with the URL. Application cannot start"
        )

# Used for posting user queries and getting their response from the RAG application
@app.post("/qna", response_model=RAGResponse, status_code=status.HTTP_200_OK)
async def read_qa(user_request: RequestQuery):
    global VECTOR_STORE
    try:
       result = await QnALoop(user_request.query, VECTOR_STORE)
       return RAGResponse(response = result)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail = "Bad request"
        )
