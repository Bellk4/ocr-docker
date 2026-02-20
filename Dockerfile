FROM vllm/vllm-openai:nightly

# 1. システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 2. pip自体のアップグレードと基本ライブラリのインストール
RUN pip install --upgrade pip setuptools wheel
RUN pip install runpod requests pillow

# 3. Transformersを最新版（GitHubメインブランチ）に入れ替え
# vLLMが持っている古いバージョンとの競合を避けるため強制再インストール
RUN pip uninstall -y transformers || true
RUN pip install git+https://github.com/huggingface/transformers.git

# 4. GLM-OCRのインストール
RUN pip install git+https://github.com/zai-org/glm-ocr.git

# 5. モデルの事前ダウンロード（ビルド時に同梱してコールドスタートを高速化）
ENV HF_HOME=/root/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('zai-org/GLM-OCR')"

EXPOSE 8080

# 起動コマンド
CMD ["vllm", "serve", "zai-org/GLM-OCR", "--allowed-local-media-path", "/", "--port", "8080"]