"""Cloud inference endpoint for the malaria cell-image classifier.

POST /api/predict with JSON {"image": "<base64 PNG or JPG>"}.

Runs the same my_model.onnx as the in-browser demo, with the same
preprocessing (resize to 50x50 RGB, pixels / 255, NHWC float32) and the
same 8-view test-time augmentation (rotations x horizontal flips).
Returns the mean P(infected) plus a 95% interval over the 8 views.
"""

import base64
import binascii
import io
import json
import math
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
from PIL import Image

SIZE = 50
MODEL_FILE = os.path.join(os.path.dirname(__file__), "my_model.onnx")
MAX_BODY = 10 * 1024 * 1024

_session = None


def get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])
    return _session


# The browser's canvas rotate(a) is clockwise for positive a (y axis points
# down); PIL's transpose() rotates counter-clockwise, so canvas angle a maps
# to PIL ROTATE_{(360 - a) % 360}. With the flip applied before the rotation,
# exactly as the canvas code composes its transforms.
_ROTATE = {
    0: None,
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def make_views(base):
    """The 8 deterministic geometric views the demo uses: 4 rotations each of
    the original and the horizontally flipped 50x50 image."""
    views = []
    for flip in (False, True):
        for angle in (0, 90, 180, 270):
            v = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else base
            t = _ROTATE[angle]
            if t is not None:
                v = v.transpose(t)
            views.append(v)
    return views


def run_tta(img):
    sess = get_session()
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    base = img.resize((SIZE, SIZE), Image.BILINEAR)
    probs = []
    for view in make_views(base):
        arr = np.asarray(view, dtype=np.float32) / 255.0
        arr = arr.reshape(1, SIZE, SIZE, 3)
        out = sess.run([out_name], {in_name: arr})[0]
        probs.append(float(out[0][0]))  # column 0 = P(infected)
    return probs


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(
            405,
            {"error": "Send a POST with JSON {\"image\": \"<base64-encoded PNG or JPG>\"}."},
        )

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._send(
                400,
                {"error": "Send a JSON body with an 'image' field containing a base64-encoded image."},
            )
            return
        try:
            data = json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, {"error": "That wasn't valid JSON. Send {\"image\": \"<base64>\"}."})
            return

        b64 = data.get("image", "") if isinstance(data, dict) else ""
        if isinstance(b64, str) and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1] if "," in b64 else ""
        if not b64 or not isinstance(b64, str):
            self._send(
                400,
                {"error": "No image found. Send {\"image\": \"<base64-encoded PNG or JPG>\"}."},
            )
            return
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            self._send(400, {"error": "The image data isn't valid base64."})
            return
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            self._send(
                400,
                {"error": "That doesn't look like an image file. Try a PNG or JPG of a single stained cell."},
            )
            return

        try:
            probs = run_tta(img)
        except Exception:
            self._send(500, {"error": "The model failed to run on that image. Try a different one."})
            return

        n = len(probs)
        mean = sum(probs) / n
        sd = math.sqrt(sum((p - mean) ** 2 for p in probs) / (n - 1)) if n > 1 else 0.0
        lo = max(0.0, mean - 1.96 * sd)
        hi = min(1.0, mean + 1.96 * sd)
        self._send(
            200,
            {
                "label": "infected" if mean >= 0.5 else "uninfected",
                "probability": round(mean, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "n_augmentations": n,
                "model": "malaria-cnn 93.2%",
            },
        )
