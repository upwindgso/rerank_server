FROM python:3.10-slim

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app /app/app
COPY models /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=true
ENV TORCH_NUM_THREADS=4

# Expose the port
EXPOSE 8000

# Run with uvicorn
CMD ["python", "app/server.py"]

