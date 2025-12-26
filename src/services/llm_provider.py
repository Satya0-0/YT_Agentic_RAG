from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import get_config
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_llm():
    return ChatGoogleGenerativeAI(
        model=get_config("model.name"),
        temperature=get_config("model.temperature"),
        timeout=get_config("model.timeout"),
        max_retries=get_config("model.max_retries"),
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )
