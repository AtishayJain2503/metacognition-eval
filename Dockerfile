# Production Dockerfile for Hermetic Benchmark Evaluation
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application source code and benchmarks
COPY nemo_eval/ ./nemo_eval/
COPY tests/ ./tests/
COPY gsm8k_config.json ./

# Default entrypoint
ENTRYPOINT ["python", "-m", "nemo_eval.cli"]
CMD ["--help"]
