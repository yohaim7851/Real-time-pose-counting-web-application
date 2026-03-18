FROM python:3.10-slim

WORKDIR /app

# OpenCV / MediaPipe 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU 전용 PyTorch 먼저 설치 (GPU 버전 대비 이미지 크기 ~3GB 절감)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 (torch는 위에서 설치했으므로 중복 설치 스킵됨)
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt

# 소스 복사 (research 디렉토리 등은 .dockerignore에서 제외)
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
