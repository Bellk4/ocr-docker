"""
RunPod Serverless handler for GLM-OCR via vLLM.

Starts vLLM HTTP server in the background, waits for it to be ready,
then forwards incoming RunPod jobs to the local OpenAI-compatible API.
"""

import os
import time
import base64
import subprocess
import threading
import logging
from io import BytesIO
from urllib.parse import urlparse
import requests
import runpod
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

VLLM_PORT = 8080
VLLM_URL = f"http://localhost:{VLLM_PORT}"
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "16384")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
SPECULATIVE_CONFIG = os.getenv(
    "SPECULATIVE_CONFIG",
    '{"method": "mtp", "num_speculative_tokens": 1}',
)
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "0").lower() in {"1", "true", "yes"}
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2000"))


def stream_output(pipe):
    """Stream vLLM logs into worker logs for easier debugging."""
    try:
        for line in pipe:
            line = line.strip()
            if line:
                log.info("[vllm] %s", line)
    except Exception as exc:
        log.exception("Error while streaming vLLM logs: %s", exc)
    finally:
        pipe.close()


def start_vllm():
    """Start vLLM as a background process with log forwarding."""
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--allowed-local-media-path", "/",
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--speculative-config", SPECULATIVE_CONFIG,
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")

    log.info("Starting vLLM: %s", " ".join(cmd))
    log.info(
        "vLLM context window configured to %s tokens (gpu_memory_utilization=%s)",
        MAX_MODEL_LEN,
        GPU_MEMORY_UTILIZATION,
    )
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        t = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        t.start()

    return process


def wait_for_vllm(timeout=600):
    """Wait for vLLM to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_URL}/health", timeout=2)
            if r.status_code == 200:
                log.info("vLLM is ready")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not start within {timeout}s")


def _extract_image_url(content_part):
    """Return image URL string from an OpenAI content part."""
    if not isinstance(content_part, dict):
        return None
    if content_part.get("type") != "image_url":
        return None

    image_url = content_part.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        return image_url.get("url")
    return None


def _set_image_url(content_part, new_url):
    """Update image_url field while preserving OpenAI-compatible shape."""
    image_url = content_part.get("image_url")
    if isinstance(image_url, dict):
        image_url["url"] = new_url
    else:
        content_part["image_url"] = {"url": new_url}


def _read_image_bytes(url):
    """Read image bytes from http(s), file://, or absolute local path."""
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content

    if parsed.scheme == "file":
        with open(parsed.path, "rb") as f:
            return f.read()

    if parsed.scheme == "" and url.startswith("/"):
        with open(url, "rb") as f:
            return f.read()

    raise ValueError(f"Unsupported image URL scheme: {parsed.scheme or 'relative-path'}")


def _resize_image_to_data_url(image_bytes, max_side):
    """Resize image if needed and return a data URL (or None if unchanged)."""
    with Image.open(BytesIO(image_bytes)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_side:
            return None, (width, height), (width, height)

        ratio = max_side / float(longest)
        new_size = (
            max(1, int(width * ratio)),
            max(1, int(height * ratio)),
        )
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        out = BytesIO()

        has_alpha = "A" in resized.getbands()
        if has_alpha:
            resized.save(out, format="PNG", optimize=True)
            mime = "image/png"
        else:
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            resized.save(out, format="JPEG", quality=90, optimize=True)
            mime = "image/jpeg"

        encoded = base64.b64encode(out.getvalue()).decode("ascii")
        return f"data:{mime};base64,{encoded}", (width, height), new_size


def preprocess_images(job_input, job_id):
    """
    Resize image_url content parts to reduce visual token usage.
    Disabled when MAX_IMAGE_SIDE <= 0.
    """
    if MAX_IMAGE_SIDE <= 0:
        return

    messages = job_input.get("messages")
    if not isinstance(messages, list):
        return

    seen = 0
    resized = 0
    skipped = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            url = _extract_image_url(part)
            if not url:
                continue

            seen += 1
            try:
                image_bytes = _read_image_bytes(url)
                data_url, old_size, new_size = _resize_image_to_data_url(
                    image_bytes, MAX_IMAGE_SIDE
                )
                if data_url is None:
                    skipped += 1
                    continue

                _set_image_url(part, data_url)
                resized += 1
                log.info(
                    "Job %s: resized image %s from %sx%s to %sx%s",
                    job_id,
                    seen,
                    old_size[0],
                    old_size[1],
                    new_size[0],
                    new_size[1],
                )
            except Exception as exc:
                skipped += 1
                log.warning("Job %s: image resize skipped (%s)", job_id, exc)

    if seen:
        log.info(
            "Job %s: image preprocessing complete (seen=%s resized=%s skipped=%s max_side=%s)",
            job_id,
            seen,
            resized,
            skipped,
            MAX_IMAGE_SIDE,
        )


def handler(job):
    """
    RunPod handler. Forwards the job input directly to vLLM's
    OpenAI-compatible chat completions endpoint.

    Expected input format (same as OpenAI chat completions):
    {
        "model": "zai-org/GLM-OCR",
        "messages": [...],
        "max_tokens": 2048,
        "temperature": 0.0
    }
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return {
            "error": (
                "Invalid request format. Expected {'input': {...}} as the "
                "job payload."
            )
        }

    job_input = dict(job_input)
    log.info("Job %s: received request", job_id)

    # Set model default if not provided.
    if "model" not in job_input:
        job_input["model"] = MODEL_NAME

    # vLLM chat completions requires messages.
    if "messages" not in job_input:
        log.error("Job %s: missing required 'messages' field", job_id)
        return {
            "error": "Input must contain a 'messages' field for Chat Completions API."
        }

    preprocess_images(job_input, job_id)

    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=job_input,
            timeout=600,
        )

        if response.status_code != 200:
            log.error(
                "Job %s: vLLM returned %s with body: %s",
                job_id,
                response.status_code,
                response.text,
            )

        response.raise_for_status()
        result = response.json()
        log.info("Job %s: completed", job_id)
        return result
    except requests.exceptions.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" | response_body={exc.response.text}"
        log.error("Job %s: failed - %s%s", job_id, exc, detail)
        raise


if __name__ == "__main__":
    vllm_process = start_vllm()
    wait_for_vllm()
    runpod.serverless.start({"handler": handler})
