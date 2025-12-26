# Necessary Imports
from src.state import *
from yt_dlp import YoutubeDL
import re
import uuid
import os
from  src.services.nlp_transformers import get_transcriber

# Node-1: Downloading YouTube Video

def clean_filename(title: str) -> str:
    """YouTube Video URL cleaning function"""
    forbidden_chars = r'[<>:"/\\|?*]'
    cleaned_title = re.sub(forbidden_chars, '_', title)
    cleaned_title2 = re.sub(' ', '_', cleaned_title)
    return re.sub(r'__+', '_', cleaned_title2).strip('_')


def node1_video_download(state: State) -> dict:
    """ Downloads the YouTube video to the local path with a "clean" title"""

    # Generating a unique, and safe temporary filename using 'uuid'
    temp_filename_base = str(uuid.uuid4())

    # Extracting Video Info (Title and Extension)
    with YoutubeDL({}) as yt:
        info = yt.extract_info(state.youtubeURL, download=False)
        raw_title = info.get("title")
        extension = info.get("ext")

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
        print(f"Error renaming file from {temp_path} to {final_path}: {e}")
        raise

    print("Node-1 Executed!")
    return {"video_details": target_filename}


# Node-2: Video Transcription using a Local Model (OpenAI's Whisper)

def node2_transcription(state: State) -> dict:
    """Transcribes the YouTube video"""
    # Defining the local path for the video
    audio_path = os.path.join(state.local_path, state.video_details)
    transcriber = get_transcriber()
    text_out = transcriber(audio_path)

    # Final Text to be stored in Vector DB
    transcribed_text = text_out['text']
    print("Node-2 Executed!")

    return {"transcription": transcribed_text}


# Node-3: Clean up; delete the downloaded YouTube Video

def node3_clean_up(state: State) -> None:
    """Deletes the transcribed YouTube video"""
    try:
        target_destination = os.path.join(state.local_path, state.video_details)
        os.remove(target_destination)
        print("Node_CleanUp Executed. Video Deleted!")
    except Exception as e:
        print(f"Error occurred in file. Path not found: {e}")
    return None
