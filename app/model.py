from __future__ import annotations

"""Model layer.

For the championship MVP we ship a deterministic, explainable placeholder model:
- It computes simple image statistics (brightness, saturation proxy) to choose a class.
- This makes the demo stable and repeatable offline.

Later you can swap this with a real model (ONNX/PyTorch) without changing the API.
"""

import io
from dataclasses import dataclass

from PIL import Image


LABELS = [
    "Healthy",
    "Leaf spot",
    "Yellowing",
    "Pest damage",
    "Mold",
]


@dataclass
class Prediction:
    label: str
    confidence: float


def _image_stats(img: Image.Image) -> tuple[float, float, float]:
    # returns mean R,G,B in [0..255]
    img = img.convert("RGB").resize((128, 128))
    pixels = list(img.getdata())
    n = len(pixels)
    r = sum(p[0] for p in pixels) / n
    g = sum(p[1] for p in pixels) / n
    b = sum(p[2] for p in pixels) / n
    return r, g, b


def predict_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    r, g, b = _image_stats(img)

    # Simple heuristics to keep demo consistent:
    # - If green dominates -> healthy
    # - If red/blue high with lower green -> spot/mold
    # - If overall bright and low green dominance -> yellowing
    # - Otherwise -> pests
    green_ratio = g / max(1.0, (r + g + b))
    brightness = (r + g + b) / 3.0

    if green_ratio > 0.38 and g > r and g > b:
        label = "Healthy"
        confidence = 0.86
    elif brightness > 160 and green_ratio < 0.34:
        label = "Yellowing"
        confidence = 0.80
    elif b > g and brightness < 120:
        label = "Mold"
        confidence = 0.78
    elif r > g and brightness < 140:
        label = "Leaf spot"
        confidence = 0.76
    else:
        label = "Pest damage"
        confidence = 0.74

    return {"label": label, "confidence": float(confidence)}
