from src.config import get_config
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from pathlib import Path
from transformers import pipeline

# Using a global variable to download the model only once
_embedding_object = None
_transcription = None
_text_splitter_object = None

def get_sentence_transformer():
    global _embedding_object
    if _embedding_object is None:
        _embedding_object = HuggingFaceEmbeddings(model_name=get_config("sentence_transformer.name"))
    return _embedding_object

# Update: Using Groq API for Fast Inference instead of Local Model (OpenAI's Whisper)
def get_transcription(file_loc: str):
    global _transcription
    if _transcription is None:
        client = Groq()
        _transcription = client.audio.transcriptions.create( 
            file = Path(file_loc), 
            model="whisper-large-v3-turbo", 
            language="en", 
            temperature=0.0, 
            response_format="verbose_json", 
            timestamp_granularities = ["segment"]
            )
    return _transcription

def get_text_splitter():
    global _text_splitter_object
    if _text_splitter_object is None:
        _text_splitter_object = pipeline()
    return _text_splitter_object