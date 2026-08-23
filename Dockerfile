FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installing 'ffmpeg', to be used in video transacrition
# Note: 'ffmpeg' is a system tool which requires installation using 'ffmpeg.exe' and cannot be installed vis 'requirements.txt'
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setting the working directory to the project root
WORKDIR /app

# Installing lightweight CPU-only PyTorch directly
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installing all necessary requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MiniLM is used locally in the application. Hence, downloading it with the image so save boot-up time
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Copying all project files into /app
COPY . .

EXPOSE 8080

CMD exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8080}