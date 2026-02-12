FROM vllm/vllm-openai:nightly

# git is needed for pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Install newer Transformers so GLM-OCR is recognized
RUN pip uninstall -y transformers || true \
 && pip install -U git+https://github.com/huggingface/transformers.git

# Pre-download model weights into the image so cold starts don't hit HuggingFace
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"
ENV HF_HUB_OFFLINE=1

# Persist vLLM compile cache on network volume to speed up cold starts
ENV VLLM_CACHE_ROOT=/runpod-volume/vllm-cache

EXPOSE 8080

ENTRYPOINT []
CMD ["vllm", "serve", "zai-org/GLM-OCR", "--allowed-local-media-path", "/", "--port", "8080"]
