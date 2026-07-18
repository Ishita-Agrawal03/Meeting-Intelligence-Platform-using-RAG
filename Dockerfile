# Backend image — runs the FastAPI app via uvicorn.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch FIRST, from PyTorch's own CPU index. This
# satisfies the torch==2.12.1 pin below WITHOUT pulling in the ~2GB of
# CUDA/GPU libraries (nvidia_cublas, nvidia_cudnn, triton, etc.) that
# the default PyPI wheel bundles — none of which do anything useful in
# a container with no GPU. This alone cuts build time from over an
# hour down to a few minutes.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.1

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p uploads vector_store data

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]