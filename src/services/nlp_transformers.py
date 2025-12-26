from src.config import get_config
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Using a global variable to download the model only once
_embedding_object = None
_transcribe_object = None
_text_splitter_object = None

def get_sentence_transformer():
    global _embedding_object
    if _embedding_object is None:
        _embedding_object = HuggingFaceEmbeddings(model_name=get_config("sentence_transformer.name"))
    return _embedding_object

def get_transcriber():
    global _transcribe_object
    if _transcribe_object is None:
        _transcribe_object = pipeline(
            model= get_config("transcriber.name"),
            framework = get_config("transcriber.framework"),
            device = get_config("transcriber.device"),
            return_timestamps= get_config("transcriber.timestamps"))
    return _transcribe_object

def get_text_splitter():
    global _text_splitter_object
    if _text_splitter_object is None:
        _text_splitter_object = pipeline()
    return _text_splitter_object