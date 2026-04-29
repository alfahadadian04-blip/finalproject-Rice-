from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_FILENAME = "yolo11n-cls.pt"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME

model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    # Prefer a colocated model file; fallback to Ultralytics auto-download by name.
    model_source = str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_FILENAME
    model = YOLO(model_source)
    yield
    model = None


app = FastAPI(
    title="Rice Classifier API",
    version="2026.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    content_type = (image.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    result = model.predict(pil_image, verbose=False)[0]
    probs = result.probs
    if probs is None:
        raise HTTPException(status_code=500, detail="Model did not return probabilities")

    names = result.names
    all_scores = {
        names[idx]: float(probs.data[idx].item())
        for idx in range(len(probs.data))
    }

    top_idx = int(probs.top1)
    label = names[top_idx]
    confidence = float(probs.top1conf.item())

    return {
        "label": label,
        "confidence": confidence,
        "all_scores": all_scores,
    }
