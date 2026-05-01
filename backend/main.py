from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL_FILENAME = "yolo11n-cls.pt"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class ModelSingleton:
    _instance: YOLO | None = None

    @classmethod
    def load(cls) -> YOLO:
        if cls._instance is None:
            model_source = str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_FILENAME
            cls._instance = YOLO(model_source)
        return cls._instance

    @classmethod
    def get(cls) -> YOLO:
        if cls._instance is None:
            raise RuntimeError("Model is not loaded")
        return cls._instance

    @classmethod
    def release(cls) -> None:
        cls._instance = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    ModelSingleton.load()
    yield
    ModelSingleton.release()


app = FastAPI(
    title="WMSU Rice Disease Detection API",
    version="2026.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
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

    try:
        model = ModelSingleton.get()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prediction = model.predict(pil_image, verbose=False)[0]
    probs = prediction.probs
    if probs is None:
        raise HTTPException(status_code=500, detail="Model did not return probabilities")

    names = prediction.names
    all_scores = {names[idx]: float(probs.data[idx].item()) for idx in range(len(probs.data))}

    top_index = int(probs.top1)
    return {
        "label": names[top_index],
        "confidence": float(probs.top1conf.item()),
        "all_scores": all_scores,
    }


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_root() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        requested = FRONTEND_DIST / full_path
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
