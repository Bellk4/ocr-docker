"""
ローカルテスト用モックサーバー（GPU不要）

vLLM の OpenAI 互換 API と RunPod のジョブ API を模倣します。
実際の OCR は行わず、ダミーレスポンスを返します。

起動方法:
    pip install fastapi uvicorn pillow requests
    python mock_server.py

テスト方法:
    # OpenAI 互換 API
    curl -X POST http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"zai-org/GLM-OCR","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}},{"type":"text","text":"OCRしてください"}]}]}'

    # RunPod 互換 API
    curl -X POST http://localhost:8080/run \
        -H "Content-Type: application/json" \
        -d '{"input":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}}'
"""

import time
import uuid
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mock_server")

app = FastAPI(title="GLM-OCR Mock Server")

MODEL_NAME = "zai-org/GLM-OCR"

MOCK_OCR_TEXT = (
    "## モックOCR結果\n\n"
    "これはローカルテスト用のモックサーバーです。\n"
    "実際の OCR 処理は行われていません。\n\n"
    "| 列1 | 列2 |\n"
    "|-----|-----|\n"
    "| サンプルデータ | 123 |\n"
)


def _make_chat_response(content: str) -> dict[str, Any]:
    """OpenAI 互換のチャットレスポンスを生成する。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


# ─────────────────────────── Health check ───────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock",
            }
        ],
    }


# ─────────────────────── OpenAI 互換 API ────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # 画像 URL を抽出してログに出力
    image_urls = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url_info = part.get("image_url", {})
                    if isinstance(url_info, dict):
                        image_urls.append(url_info.get("url", ""))
                    elif isinstance(url_info, str):
                        image_urls.append(url_info)

    log.info("Chat completions request: model=%s, images=%d",
             body.get("model"), len(image_urls))
    for i, url in enumerate(image_urls):
        log.info("  Image[%d]: %s", i, url[:80])

    response = _make_chat_response(MOCK_OCR_TEXT)
    return JSONResponse(content=response)


# ──────────────────────── RunPod 互換 API ───────────────────────────

@app.post("/run")
async def runpod_run(request: Request):
    body = await request.json()
    job_input = body.get("input", {})
    log.info("RunPod /run request: %s", str(job_input)[:200])

    job_id = f"job-{uuid.uuid4().hex[:8]}"
    return JSONResponse(content={
        "id": job_id,
        "status": "COMPLETED",
        "output": {
            "markdown": MOCK_OCR_TEXT,
            "layout_json": None,
            "pages": 1,
        },
    })


@app.post("/runsync")
async def runpod_runsync(request: Request):
    return await runpod_run(request)


@app.get("/status/{job_id}")
def job_status(job_id: str):
    return JSONResponse(content={
        "id": job_id,
        "status": "COMPLETED",
        "output": {
            "markdown": MOCK_OCR_TEXT,
            "pages": 1,
        },
    })


# ────────────────────────────── Main ────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("GLM-OCR モックサーバー起動中...")
    log.info("  OpenAI API : http://localhost:8080/v1/chat/completions")
    log.info("  RunPod API : http://localhost:8080/run")
    log.info("  ヘルスチェック: http://localhost:8080/health")
    log.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
