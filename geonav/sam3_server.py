"""SAM3 service. Run this in the sam3 conda env, not the Habitat Python 3.9 env.

    conda activate sam3
    cd GeoVLN
    python -m geonav.sam3_server --checkpoint /root/data1/models/sam3_weights/sam3.pt --port 8001
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from geonav._bootstrap import install

install()

from build_semantic_map import load_processor, segment_prompts  # noqa: E402


def _png(mask):
    import numpy as np
    from PIL import Image

    image = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _segment(processor, image, prompts, min_score, limit):
    import numpy as np

    results = []
    for prompt, masks, scores in segment_prompts(processor, image, prompts):
        if scores is None:
            order = range(len(masks))
            score_of = lambda index: 1.0  # noqa: E731
        else:
            order = list(np.argsort(-np.reshape(scores, -1)))
            flat = np.reshape(scores, -1)

            def score_of(index, flat=flat):
                return float(flat[index])

        kept_masks = []
        kept_scores = []
        for index in order:
            score = score_of(index)
            if score < min_score:
                continue
            kept_masks.append(_png(masks[index]))
            kept_scores.append(round(score, 4))
            if len(kept_masks) >= limit:
                break
        results.append({"prompt": prompt, "masks": kept_masks, "scores": kept_scores})
    return results


def serve(processor, host, port, min_score, limit):
    from PIL import Image

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"ok": False})
                return
            self._send(200, {"ok": True})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/segment":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                image = Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")
                prompts = [str(item) for item in body.get("prompts") or [] if str(item).strip()]
                results = _segment(processor, image, prompts, min_score, limit)
            except Exception as exc:  # noqa: BLE001 - report the failure to the client
                self._send(500, {"error": str(exc)})
                return
            self._send(200, {"results": results})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s\n" % (fmt % args))

        def _send(self, status, payload):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"SAM3 listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="SAM3 mask service for the geonav loop")
    parser.add_argument("--checkpoint", default="/root/data1/models/sam3_weights/sam3.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=5, help="masks kept per prompt")
    args = parser.parse_args()
    processor = load_processor(args.checkpoint)
    serve(processor, args.host, args.port, args.min_score, args.limit)


if __name__ == "__main__":
    main()
