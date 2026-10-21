#!/usr/bin/env python3
"""OpenAI 兼容的 VLM 调用。文本和图像走同一个入口，解析失败时返回 None。

给本地 vLLM / SGLang 或云端兼容接口用，不绑定某一家 SDK。

环境变量（命令行参数优先）：

- ``VLM_BASE_URL``  默认 ``http://127.0.0.1:8000/v1``
- ``VLM_API_KEY``   默认 ``EMPTY``
- ``VLM_MODEL``     默认 ``Qwen2.5-VL-7B-Instruct``
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def extract_json(text: str) -> dict | None:
    """取回复里第一个完整的 JSON 对象。围栏、前后说明都忽略。"""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def _data_url(image) -> str:
    if isinstance(image, (str, Path)):
        path = Path(image)
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(
            suffix, "image/jpeg"
        )
    elif isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        mime = "image/jpeg"
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        raw = buffer.getvalue()
        mime = "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class VLMClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=180):
        self.base_url = (base_url or os.environ.get("VLM_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else (os.environ.get("VLM_API_KEY") or "EMPTY")
        self.model = model or os.environ.get("VLM_MODEL") or "Qwen2.5-VL-7B-Instruct"
        self.timeout = timeout

    def warmup(self, keep_alive: str = "30m", timeout: int = 900) -> None:
        """让 Ollama 先把权重装进显存。装载用单独的超时，之后的请求才受 self.timeout 限制。"""
        origin = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        url = origin.rstrip("/") + "/api/generate"
        payload = {"model": self.model, "keep_alive": keep_alive, "stream": False}
        print(f"loading {self.model}; per-request timeout of {self.timeout}s starts after this", flush=True)
        started = time.perf_counter()
        status, detail = self._post_url(url, payload, timeout)
        if status == 400:
            payload["prompt"] = "ok"
            payload["options"] = {"num_predict": 1}
            status, detail = self._post_url(url, payload, timeout)
        if status == 404 and "not found" in detail.lower() and "model" not in detail.lower():
            print("no Ollama preload endpoint, continuing", file=sys.stderr)
            return
        if status != 200:
            raise SystemExit(f"model {self.model} did not load: {detail}")
        print(f"model ready in {time.perf_counter() - started:.0f}s", flush=True)

    def complete(self, text: str, images=None, max_tokens=1024, temperature=0.0) -> str | None:
        """返回模型文本。网络错误、HTTP 错误、空回复都返回 None。"""
        content = []
        for image in images or []:
            content.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
        content.append({"type": "text", "text": text})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = self._post("/chat/completions", payload)
        if not body:
            return None
        try:
            message = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print("VLM response has no choices[0].message.content", file=sys.stderr)
            return None
        if isinstance(message, list):
            message = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        text_out = (message or "").strip()
        return text_out or None

    def complete_json(self, text: str, images=None, max_tokens=1024, temperature=0.0, retries=1) -> dict | None:
        """要一个 JSON 对象。解析不出来就带上失败说明再问一次，仍失败返回 None。"""
        prompt = text
        for attempt in range(retries + 1):
            raw = self.complete(prompt, images=images, max_tokens=max_tokens, temperature=temperature)
            parsed = extract_json(raw or "")
            if parsed is not None:
                return parsed
            if attempt == retries:
                return None
            prompt = (
                text
                + "\n\nThe previous reply was not a single JSON object. "
                + "Reply again with only the JSON object, no markdown fence and no explanation."
            )
        return None

    def _post(self, path: str, payload: dict) -> dict | None:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except TimeoutError:
            print(f"VLM request timed out after {self.timeout}s", file=sys.stderr)
            return None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            print(f"VLM HTTP {exc.code}: {detail}", file=sys.stderr)
            return None
        except urllib.error.URLError as exc:
            print(f"VLM request failed: {exc.reason}", file=sys.stderr)
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("VLM returned a body that is not JSON", file=sys.stderr)
            return None

    def _post_url(self, url: str, payload: dict, timeout: int) -> tuple[int, str]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")[:400]
        except TimeoutError:
            return 0, f"timed out after {timeout}s"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            return exc.code, f"HTTP {exc.code}: {detail}"
        except urllib.error.URLError as exc:
            return 0, f"request failed: {exc.reason}"
