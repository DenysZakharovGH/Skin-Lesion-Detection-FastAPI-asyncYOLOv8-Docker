FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
 && pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y numpy \
 && pip install --no-cache-dir numpy==1.26.4

COPY app/ app/
COPY frontend/ frontend/
COPY models/ models/
COPY tests/ tests/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--workers", "4", "--host", "0.0.0.0", "--port", "8000"]