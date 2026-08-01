from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class VideoQA(BaseModel):
    url: str
    query: str

@app.put("/")
def read_url(video_qa: VideoQA):
    return {"message": f"Received URL: {video_qa.url}"}

@app.put("/qa")
def read_qa(video_qa: VideoQA):
    return {"message": f"Received Query: {video_qa.query}"}

@app.get("/qa")
def get_qa():
    return {"message": "This endpoint will return the Q&A results."}