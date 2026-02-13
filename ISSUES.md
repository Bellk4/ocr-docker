# Known Issues

## 1. Requests from RunPod Admin UI queue but never execute

**Status:** Fixed

RunPod Serverless uses a job queue protocol (`/run`, `/runsync`), not HTTP proxying. Containers must use the `runpod` Python SDK with `runpod.serverless.start({"handler": handler})`. Running `vllm serve` alone gives you a working HTTP server that RunPod cannot route jobs to.

**Fix:** Added `handler.py` that starts vLLM as a subprocess, waits for it to be healthy, then uses `runpod.serverless.start()` to receive jobs and forward them to the local vLLM API. Dockerfile CMD changed from `vllm serve ...` to `python3 -u /handler.py`.

## 2. Remote test image URL returns 403

**Status:** Workaround available

The example image URL from the GLM-OCR docs (`https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/...`) returns 403 Forbidden when fetched from RunPod workers. The Chinese CDN likely blocks non-Chinese IPs.

**Workaround:** Use a different public image URL for testing, e.g. a Wikipedia image.

## 3. Shell quoting corrupts JSON in SSH sessions

**Status:** Workaround available

Pasting multi-line curl commands into RunPod SSH sessions causes line breaks and spaces to be inserted into the JSON body, resulting in `Missing 'type' field in multimodal part` errors (400 Bad Request).

**Workaround:** Use Python `urllib.request` instead of curl to avoid shell quoting issues, or write JSON to a file first and use `curl -d @file.json`.

## 4. Cold start takes ~15-30 seconds (with compile cache)

**Status:** Mitigated

Initial cold start without compile cache was ~126 seconds. After adding `VLLM_CACHE_ROOT=/runpod-volume/vllm-cache` with a network volume, cold start dropped to ~15 seconds.

Without the network volume, every cold start recompiles CUDA graphs (~2 minutes at 100% CPU).

## Build issues encountered and resolved

These are documented for reference — all fixed in the current Dockerfile.

| Issue | Cause | Fix |
|-------|-------|-----|
| `git: not found` during build | Base image lacks git | `apt-get install git` |
| `python: not found` during build | Base image only has `python3` | Use `python3` explicitly |
| `huggingface-cli: not found` | CLI binary not on PATH after upgrade | Use `python3 -c "from huggingface_hub import snapshot_download; ..."` |
| `vllm: unrecognized arguments: serve` | Base image has `ENTRYPOINT ["vllm"]`, CMD doubled it | Added `ENTRYPOINT []` |
| `cuda>=12.9 unsatisfied` | 4090 RunPod workers have older drivers | Set CUDA version to 12.9+ in RunPod settings to filter compatible GPUs |
| `transformers<5` pip warning | vLLM pins transformers<5, GLM-OCR needs v5+ | Safe to ignore — vLLM works fine |
