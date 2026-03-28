FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installing 'ffmpeg', to be used in video transacrition
# 'ffmpeg' is a system tool which requires installation using 'ffmpeg.exe' and cannot be installed vis 'requirements.txt'
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app/src

COPY requirements.txt .
# Installing "CPU-only" version of PyTorch since Whisper is used basis CPU only
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Run requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Whisper and MiniLM are used locally in the application. Hence, downloading them with the image
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-small')"
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "main.py"]