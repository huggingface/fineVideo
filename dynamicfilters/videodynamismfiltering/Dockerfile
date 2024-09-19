# Use the official Python image with necessary packages
FROM python:3.9-slim

# Install ffmpeg and other dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the Python script into the container
COPY check_static.py .

# Install Python dependencies
RUN pip install boto3 ffmpeg-python

# Command to run the script
CMD ["python", "check_static.py"]
