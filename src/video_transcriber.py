# Necessary Imports
from yt_dlp import YoutubeDL
import re
import uuid
import os
import sys
from  src.services.nlp_transformers import get_transcription, get_sentence_transformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import get_config
import logging

logger = logging.getLogger(__name__)

# Global Variables
# GLOBAL_VECTOR_STORE = None


# Downloading YouTube Video, Transcribing it and deleting it

async def step1_video_download(youtubeURL: str) -> bool:
    """ Downloads the YouTube video to the local path with a "clean" title"""
    
    # Used to store Transcription of the video

    logger.info(f"Starting video download for URL: {youtubeURL}")

    # Generating a unique, and safe temporary filename using 'uuid'
    temp_filename_base = str(uuid.uuid4())

    # Extracting Video Info (Title and Extension)
    with YoutubeDL({}) as yt:
        info = yt.extract_info(youtubeURL, download=False)
        raw_title = info.get("title")
        extension = info.get("ext")
        duration = info.get("duration")

    if duration > 1800:  # If video is longer than 15 minutes -> do not enter the LangGraph Loop
        logger.error("Error! Video is too long for demo, i.e., >20mins \nCannot proceed!!")
        return False
    
    else:
        # Using "Temporary" name to download the file
        local_path = get_config("path")
        temp_download_name = temp_filename_base + "." + extension
        temp_path = os.path.join(local_path, temp_download_name)

        # Actual download (using Temporary Name and Path)
        yt_opts = {
            'format': 'bestaudio',
            'outtmpl': temp_path
        }

        # Changing path for `os's current directory` -> this is where the download takes place Locally
        os.chdir(local_path)

        with YoutubeDL(yt_opts) as yt:
            yt.download([youtubeURL])
        logger.info("Step-1_Phase-1: Video_Download Executed!")
        # """Transcribes the YouTube video"""
        transcription_obj = get_transcription(temp_path)
        
        # Final Text to be stored in Vector DB
        transcription = transcription_obj.text

        logger.info("Step-1_Phase-2: Video Transcribed Successfully!")

        # """Delete the transcribed YouTube video"""
        try:
            os.remove(temp_path)
            logger.info("Step-1_Phase-3: CleanUp Executed. Video Deleted!")
        except Exception as e:
            logger.error(f"Error occurred in file. Video couldn't be deleted. Path not found: {e}")
        return transcription


# Step-2: Creating a VectorDB Collection
async def step2_InitiateVectorDB(transcription: str) -> None:
    """Creates the ChromaDB collection for the video transcription"""

    # Splitting the transcribed text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=get_config("text_splitter.chunk_size"),
        chunk_overlap=get_config("text_splitter.overlap"),
        length_function=get_config("text_splitter.length"),
        is_separator_regex=get_config("text_splitter.is_separator_regex")
    )

    docs = text_splitter.create_documents([transcription])
    sentence_transformer = get_sentence_transformer()
    vector_store = Chroma(
        collection_name = get_config("vectorDB_collection_name"),
        embedding_function = sentence_transformer
    )

    # Adding documents to ChromaDB Collection - "YT_RAG"

    document_ids = list(f"id_{x}" for x in range(len(docs)))

    vector_store.add_documents(documents=docs, ids=document_ids)

    logger.info("Step-2: VectorDB Executed! ChromaDB is ready.")

    return vector_store