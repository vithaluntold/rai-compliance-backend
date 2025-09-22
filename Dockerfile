FROM python:3.11-slim

# Set environment variables to force binary packages
ENV PIP_ONLY_BINARY=spacy,thinc,cymem,murmurhash,wasabi
ENV PIP_PREFER_BINARY=1
ENV PIP_NO_COMPILE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with explicit wheel preference
RUN pip install --upgrade pip && \
    pip install --only-binary=all spacy==3.7.6 && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]