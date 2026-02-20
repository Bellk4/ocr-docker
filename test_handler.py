"""
handler.py の GPU 不要部分の単体テスト

テスト対象:
  - 画像 URL の抽出ロジック
  - リクエストペイロードの正規化
  - 画像リサイズロジック（Pillowのみ使用）
  - vLLM へのリクエスト形式の検証（モックサーバー使用）

実行方法:
    .venv\\Scripts\\python.exe test_handler.py
"""

import sys
import os
import json
import base64
import subprocess
import time
import threading
import requests
from io import BytesIO

# PIL を先にロードしてプラグイン（JPEG等）を登録しておく
# これを行わないと handler.py が PIL を別インスタンスでロードし JPEG が認識されない
from PIL import Image, JpegImagePlugin, PngImagePlugin  # noqa: F401

# handler.py を直接インポート（vllm/runpod がなくても import できる部分だけ使う）
sys.path.insert(0, os.path.dirname(__file__))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, condition))


# ─────────────────────────────────────────────
# 1. handler.py の関数を安全にインポート
# ─────────────────────────────────────────────
print("\n[1] handler.py の関数インポート")
try:
    from unittest.mock import MagicMock, patch

    # runpod と vllm がなくてもインポートできるようにモック化
    with patch.dict("sys.modules", {
        "runpod": MagicMock(),
        "vllm": MagicMock(),
    }):
        import importlib
        import handler as h
    check("handler.py インポート成功", True)
except Exception as e:
    check("handler.py インポート成功", False, str(e))
    h = None


# ─────────────────────────────────────────────
# 2. 画像URL抽出ロジック
# ─────────────────────────────────────────────
print("\n[2] 画像URL抽出ロジック (_extract_job_image_and_prompt)")
if h:
    # パターン1: シンプルな url キー
    image, prompt = h._extract_job_image_and_prompt({"url": "https://example.com/test.png"})
    check("url キーから画像URL抽出", image == "https://example.com/test.png")

    # パターン2: messages 形式
    payload = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                {"type": "text", "text": "OCRしてください"}
            ]}
        ]
    }
    image, prompt = h._extract_job_image_and_prompt(payload)
    check("messages 形式から画像URL抽出", image == "https://example.com/img.jpg")
    check("messages 形式からプロンプト抽出", prompt == "OCRしてください")

    # パターン3: 画像なし
    image, prompt = h._extract_job_image_and_prompt({"messages": []})
    check("画像なしの場合 None を返す", image is None)
else:
    print("  (スキップ: handler.py インポート失敗)")


# ─────────────────────────────────────────────
# 3. 画像リサイズロジック
# ─────────────────────────────────────────────
print("\n[3] 画像リサイズロジック (_resize_image_to_data_url)")
if h:
    from PIL import Image
    import struct, zlib

    def make_test_image(width, height):
        img = Image.new("RGB", (width, height), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()  # getvalue() は位置に関係なく全バイトを返す

    # 小さい画像（リサイズ不要）
    small_bytes = make_test_image(100, 100)
    data_url, old_size, new_size = h._resize_image_to_data_url(small_bytes, 2000)
    check("小さい画像はリサイズしない", data_url is None, f"{old_size} → {new_size}")

    # 大きい画像（リサイズ必要）
    large_bytes = make_test_image(4000, 3000)
    data_url, old_size, new_size = h._resize_image_to_data_url(large_bytes, 2000)
    check("大きい画像はリサイズされる", data_url is not None, f"{old_size} → {new_size}")
    check("リサイズ後はMax辺が2000以下", max(new_size) <= 2000, str(new_size))
    check("data URL 形式が正しい", data_url.startswith("data:image/"))
else:
    print("  (スキップ: handler.py インポート失敗)")


# ─────────────────────────────────────────────
# 4. モックサーバーとの疎通確認
# ─────────────────────────────────────────────
print("\n[4] モックサーバーとの疎通確認")

# モックサーバーをサブプロセスで起動
venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
if not os.path.exists(venv_python):
    venv_python = sys.executable

mock_proc = subprocess.Popen(
    [venv_python, os.path.join(os.path.dirname(__file__), "mock_server.py")],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# 起動待機
time.sleep(2)

try:
    # ヘルスチェック
    r = requests.get("http://localhost:8080/health", timeout=5)
    check("ヘルスチェック (GET /health)", r.status_code == 200)

    # OpenAI 互換 API
    payload = {
        "model": "zai-org/GLM-OCR",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}},
                {"type": "text", "text": "OCRしてください"}
            ]}
        ],
        "max_tokens": 2048,
        "temperature": 0.0
    }
    r = requests.post("http://localhost:8080/v1/chat/completions", json=payload, timeout=5)
    check("chat/completions ステータス 200", r.status_code == 200)
    data = r.json()
    check("choices フィールドが存在する", "choices" in data)
    check("content が文字列", isinstance(data["choices"][0]["message"]["content"], str))

    # RunPod 互換 API
    r2 = requests.post("http://localhost:8080/run", json={"input": {"url": "https://example.com/test.png"}}, timeout=5)
    check("RunPod /run ステータス 200", r2.status_code == 200)
    data2 = r2.json()
    check("status が COMPLETED", data2.get("status") == "COMPLETED")
    check("output.markdown が存在する", "markdown" in data2.get("output", {}))

except Exception as e:
    check("モックサーバー疎通", False, str(e))
finally:
    mock_proc.terminate()


# ─────────────────────────────────────────────
# 結果サマリー
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"結果: {passed}/{total} 件合格")
if passed == total:
    print("[SUCCESS] Local test passed")
    print("Next step: Deploy Docker image to RunPod")
else:
    print("[FAILURE] Some tests failed. Please fix the issues.")
print("=" * 50)
sys.exit(0 if passed == total else 1)
