FROM vllm/vllm-openai:nightly

# Install newer Transformers so GLM-OCR is recognized
RUN pip uninstall -y transformers || true \
 && pip install -U git+https://github.com/huggingface/transformers.git

# Pre-download model weights into the image so cold starts don't hit HuggingFace
ENV HF_HOME=/root/.cache/huggingface
RUN huggingface-cli download zai-org/GLM-OCR

EXPOSE 8080

CMD ["vllm", "serve", "zai-org/GLM-OCR", "--allowed-local-media-path", "/", "--port", "8080"]
