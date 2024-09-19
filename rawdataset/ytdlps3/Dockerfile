FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    && apt-get clean

# Install yt-dlp (a fork of youtube-dl with more features and better maintenance)
RUN pip install yt-dlp boto3

# Create a directory for the application
WORKDIR /app

# Copy the script into the Docker image
COPY download_and_upload.py /app/download_and_upload.py

# Set the entry point to the script
ENTRYPOINT ["python", "/app/download_and_upload.py"]
