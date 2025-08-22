# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    gfortran \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_web.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_web.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/temp /app/uploads /app/static

# Ensure model weights exist (copy if available)
# Note: Model weights should be downloaded or provided during build
RUN if [ ! -f "best_weights_PoseRAC.pth" ] && [ ! -f "new_weights.pth" ]; then \
    echo "Warning: No model weights found. Please ensure model weights are available."; \
    fi

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting PoseRAC Exercise Counter Web Service..."\n\
echo "Starting FastAPI backend on port 8000..."\n\
uvicorn app:app --host 0.0.0.0 --port 8000 &\n\
echo "Starting Streamlit frontend on port 8501..."\n\
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &\n\
echo "Both services started. Backend: http://localhost:8000, Frontend: http://localhost:8501"\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV BACKEND_URL=http://localhost:8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/app/start.sh"]