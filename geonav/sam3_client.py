"""Ask the SAM3 process for masks. This module does not import torch or sam3."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


class Sam3Client:
    def __init__(self, url="http://127.0.0.1:8001", timeout=180):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            return False
        return bool(body.get("ok"))

    def segment(self, rgb, prompts):
        """Return ``[{prompt, masks, scores}]`` or None when the service cannot answer."""
        if not prompts:
            return []
        image = np.asarray(rgb)
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]
        buffer = io.BytesIO()
        Image.fromarray(image.astype(np.uint8)).save(buffer, format="JPEG", quality=90)
        payload = {
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "prompts": list(prompts),
        }
        try:
            body = self._post("/segment", payload)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        results = []
        for item in body.get("results") or []:
            masks = []
            scores = []
            for mask_b64, score in zip(item.get("masks") or [], item.get("scores") or []):
                raw = base64.b64decode(mask_b64)
                mask = np.array(Image.open(io.BytesIO(raw))) > 0
                masks.append(mask)
                scores.append(float(score))
            if masks:
                results.append({"prompt": item.get("prompt", ""), "masks": masks, "scores": scores})
        return results

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
