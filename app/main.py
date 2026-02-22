from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.ml_model import predict_image as ml_predict
from .db import init_db, add_prediction, list_history, clear_history
from .recommendations import get_recommendations

APP_NAME = "AgroAI Web"

app = FastAPI(title=APP_NAME)

# CORS for local dev (frontend on 5173). For demo you can keep "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.getenv(
    "AGROAI_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data"),
)
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# optional: expose uploaded images (handy for demo)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.on_event("startup")
def _startup():
    init_db(os.path.join(DATA_DIR, "agroai.sqlite3"))


@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME}


@app.post("/predict")
async def predict(file: UploadFile = File(...), culture: Optional[str] = None):
    # validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Пожалуйста, загрузите изображение (image/*).")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    # Save file for history/demo
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = (file.filename or "image").replace("/", "_").replace("\\", "_")
    saved_name = f"{ts}_{safe_name}"
    saved_path = os.path.join(UPLOADS_DIR, saved_name)
    with open(saved_path, "wb") as f:
        f.write(image_bytes)

    # ML prediction
    pred = ml_predict(image_bytes)  # expects: {"label": ..., "confidence": ..., "recommendations": ... (optional)}

    # Recommendations: if ml already returns, use it; else use get_recommendations()
    tips = pred.get("recommendations")
    if not tips:
        tips = get_recommendations(pred["label"], culture=culture)

    # Persist to SQLite
    row_id = add_prediction(
        filename=saved_name,
        label=pred["label"],
        confidence=float(pred["confidence"]),
        culture=culture,
    )

    return {
        "id": row_id,
        "label": pred["label"],
        "confidence": pred["confidence"],
        "recommendations": tips,
        "image_url": f"/uploads/{saved_name}",
        "culture": culture,
    }


@app.get("/history")
def history(limit: int = 10):
    limit = max(1, min(limit, 100))
    items = list_history(limit=limit)

    # attach image urls
    for it in items:
        it["image_url"] = f"/uploads/{it['filename']}" if it.get("filename") else None

    return {"items": items}


@app.delete("/history")
def history_clear():
    clear_history()
    return {"ok": True}