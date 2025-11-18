FROM python:3.10-slim

# Base image
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Faster, cleaner Python + logs in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


# Workdir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# Copy your code and data
COPY main.py /app/main.py
COPY converted_pdfs /app/converted_pdfs

# Expose Fly's internal port
EXPOSE 8080

# Start FastAPI (bind 0.0.0.0:8080 for Fly)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]




