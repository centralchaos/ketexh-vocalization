FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /staging && chmod 777 /staging

RUN mkdir -p /app/app/models
COPY app/models/ketexh-vocalization.pth /app/app/models/

ENV PYTHONPATH=/app