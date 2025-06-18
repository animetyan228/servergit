FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr tesseract-ocr-rus \
    poppler-utils \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Используем Gunicorn для запуска
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--timeout", "300", "--workers", "1"]
