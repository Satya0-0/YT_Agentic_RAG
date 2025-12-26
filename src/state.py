from src.config import get_config
from pydantic import BaseModel
from typing import List, Optional


# Creating the State:

class State(BaseModel):
    """Defines the State of Graph"""
    class Config:
        arbitrary_types_allowed = True

    user_query: str = None
    youtubeURL: str = None
    local_path: str = get_config("path")
    video_details: str = ""
    transcription: str = ""
    vectorDB_flg: bool = False
    rewritten_query: Optional[str] = None
    rewritten_flg : bool = False
    retrieval_sync : bool = True
    documents: List[str] = []
    webResults: Optional[str] = None
    graph_output: str = ""
    graph_exit: bool = True