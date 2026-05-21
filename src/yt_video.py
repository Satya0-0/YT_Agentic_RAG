# Necessary Imports
from src.state import *
from yt_dlp import YoutubeDL
import re
import uuid
import os
from  src.services.nlp_transformers import get_transcription
import logging

logger = logging.getLogger(__name__)

# Node-1: Downloading YouTube Video

def clean_filename(title: str) -> str:
    """YouTube Video URL cleaning function"""
    forbidden_chars = r'[<>:"/\\|?*]'
    cleaned_title = re.sub(forbidden_chars, '_', title)
    cleaned_title2 = re.sub(' ', '_', cleaned_title)
    return re.sub(r'__+', '_', cleaned_title2).strip('_')


def node1_video_download(state: State) -> dict:
    """ Downloads the YouTube video to the local path with a "clean" title"""
    
    logger.info(f"Starting video download for URL: {state.youtubeURL}")

    # Generating a unique, and safe temporary filename using 'uuid'
    temp_filename_base = str(uuid.uuid4())
    # Setting a default value for demo_ok flag to True; will be set to False if any issues arise in the video download process (like video being too long, or file handling issues)
    demo_ok = True

    # Extracting Video Info (Title and Extension)
    with YoutubeDL({}) as yt:
        info = yt.extract_info(state.youtubeURL, download=False)
        raw_title = info.get("title")
        extension = info.get("ext")
        duration = info.get("duration")

    if duration > 900:  # If video is longer than 15 minutes -> exit the LangGraph Loop
        logger.error("Error! Video is too long for demo. \nExiting LangGraph Loop!!")
        demo_ok = False
        target_filename = None
    
    else:
        # Cleaning the Video Title name (to be used for downloading)
        safe_title_base = clean_filename(raw_title)
        target_filename = safe_title_base + "." + extension

        # Using "Temporary" name to download the file
        temp_download_name = temp_filename_base + "." + extension
        temp_path = os.path.join(state.local_path, temp_download_name)

        # Actual download (using Temporary Name and Path)
        yt_opts = {
            'format': 'bestaudio',
            'outtmpl': temp_path
        }

        # Changing path for `os's current directory` -> this is where the download takes place Locally
        os.chdir(state.local_path)

        with YoutubeDL(yt_opts) as yt:
            yt.download([state.youtubeURL])
            
        # Renaming the downloaded file name from "Temporary Name" to "Clean Name"
        final_path = os.path.join(state.local_path, target_filename)
        try:
            os.rename(temp_path, final_path)
        except Exception as e:
            logger.error(f"Error renaming file from {temp_path} to {final_path}: {e}")
            demo_ok = False
            target_filename = None

    logger.info("Node-1 Executed! Video Downloaded and Renamed Successfully!")
    return {"video_details": target_filename, "proceed_with_demo": demo_ok}


# Node-2 {Updated}: Video Transcription using Groq API call for Fast Inference 
# Node-2 {Previous}:Video Transcription using a Local Model (OpenAI's Whisper)

def node2_transcription(state: State) -> dict:
    """Transcribes the YouTube video"""
    # Defining the local path for the video
    audio_path = os.path.join(state.local_path, state.video_details)
    transcription = get_transcription(audio_path)
    # Final Text to be stored in Vector DB
    transcribed_text = transcription.text
    logger.info("Node-2 Executed! Video Transcribed Successfully!")

    return {"transcription": transcribed_text}


# Node-3: Clean up; delete the downloaded YouTube Video

def node3_clean_up(state: State) -> None:
    """Deletes the transcribed YouTube video"""
    try:
        target_destination = os.path.join(state.local_path, state.video_details)
        os.remove(target_destination)
        logger.info("Node3_CleanUp Executed. Video Deleted!")
    except Exception as e:
        logger.error(f"Error occurred in file. Path not found: {e}")
    return None
