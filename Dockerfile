# File: Dockerfile
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

# Install handler dependencies
RUN pip install --no-cache-dir runpod pillow requests

# Copy the RunPod serverless handler
COPY handler.py /handler.py

EXPOSE 8080

# handler.py starts vLLM as a subprocess, then listens for RunPod jobs
CMD ["python3", "-u", "/handler.py"]