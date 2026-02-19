# ---------- stage 1: build flash-attn wheel ----------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git curl ca-certificates \
    build-essential ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN python3 -m pip install -U psutil packaging

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl ca-certificates \
    build-essential ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*
    
ENV TORCH_CUDA_ARCH_LIST="8.9" \
    MAX_JOBS="2"

# попробуйте при необходимости 2.6.2 или 2.7.0
RUN python3 -m pip wheel --no-build-isolation --no-deps \
    "flash-attn==2.6.3" -w /wheels


# ---------- stage 2: runtime + notebook ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

RUN python3 -m pip install \
    "transformers==4.48.3" \
    "accelerate==1.2.1" \
    "bitsandbytes==0.48.1" \
    "safetensors>=0.4.5" \
    "sentencepiece" \
    "huggingface_hub>=0.23.0" \
    "jupyterlab>=4.0" \
    "ipykernel" \
    "ipywidgets" \
    "datasets" \
    "pandas" \
    "protobuf" \
    "diskcache" \
    "matplotlib"

COPY --from=builder /wheels /wheels
RUN python3 -m pip install /wheels/flash_attn-*.whl
RUN pip install llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

WORKDIR /notebooks
EXPOSE 8888
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
