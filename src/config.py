# Create a folder to temporarily store downloaded YouTube Videos
from pathlib import Path
import os

parent_path = Path(__file__).resolve().parent.parent
video_downloads = parent_path / "video_downloads"
video_downloads.mkdir(parents=True, exist_ok=True)
video_path = Path(video_downloads).resolve()

# Actual CONFIG dict
CONFIG = {
    "model":{
        "name":"gemini-2.5-flash-lite",
        "temperature":0,
        "timeout":10,
        "max_retries":2
    },
    "sentence_transformer" : {
        "name":"sentence-transformers/all-MiniLM-L6-v2"
    },
    "transcriber":{
        "name":"openai/whisper-small",
        "framework":"pt",
        "device":-1,
        "timestamps":True
    },
    "text_splitter": {
        "chunk_size":1000,
        "overlap":0,
        "length":len,
        "is_separator_regex":False
    },
    "path": video_path,
    "vectorDB_collection_name":"YT_RAG"
}


def get_config(string):
    keys = string.split(".")
    val = CONFIG
    for key in keys:
        if key not in val:
            raise KeyError(f"Config key '{key}' not found in {string}")
        val = val[key]
    return val