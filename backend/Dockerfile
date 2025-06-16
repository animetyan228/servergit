FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    python3 python3-pip \
    tesseract-ocr tesseract-ocr-rus \
    poppler-utils build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN grep -v gptqmodel requirements.txt > requirements_no_gptqmodel.txt && \
    pip3 install --no-cache-dir -r requirements_no_gptqmodel.txt

RUN pip3 install --no-cache-dir gptqmodel

COPY . .

CMD ["python3", "app.py"]
