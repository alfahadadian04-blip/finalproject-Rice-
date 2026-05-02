from __future__ import annotations

import io
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

MODEL_FILENAME = "yolo11n-cls.pt"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
MIN_IMAGE_WIDTH = 224
MIN_IMAGE_HEIGHT = 224
MIN_BRIGHTNESS_STD = 18.0
MIN_TOP1_CONFIDENCE = 0.68
MIN_TOP1_MARGIN = 0.12
MAX_PREDICTION_ENTROPY = 1.35


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


def build_inference_views(image: Image.Image) -> list[Image.Image]:
    """Create multiple stable views for test-time averaging."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    center_crop = image.crop((left, top, left + side, top + side)).resize(image.size, Image.Resampling.BICUBIC)
    mirrored = ImageOps.mirror(image)
    enhanced = ImageEnhance.Contrast(image).enhance(1.08)
    return [image, center_crop, mirrored, enhanced]


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
        opened_image = Image.open(io.BytesIO(raw_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    normalized_image = ImageOps.exif_transpose(opened_image)
    exif = normalized_image.getexif()
    camera_make = str(exif.get(271, "")).strip()
    camera_model = str(exif.get(272, "")).strip()
    has_camera_metadata = bool(camera_make or camera_model)

    pil_image = normalized_image.convert("RGB")
    width, height = pil_image.size
    has_low_resolution = width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT

    grayscale = pil_image.convert("L")
    pixels = list(grayscale.getdata())
    if not pixels:
        raise HTTPException(status_code=400, detail="Invalid image data")
    mean_value = sum(pixels) / len(pixels)
    variance = sum((pixel - mean_value) ** 2 for pixel in pixels) / len(pixels)
    if variance**0.5 < MIN_BRIGHTNESS_STD:
        raise HTTPException(
            status_code=400,
            detail="Image quality is too low. Use a clearer, well-lit photo.",
        )

    try:
        model = ModelSingleton.get()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    inference_views = build_inference_views(pil_image)
    predictions = model.predict(inference_views, verbose=False)
    if not predictions:
        raise HTTPException(status_code=500, detail="Model did not return predictions")

    names = predictions[0].names
    class_count = len(names)
    score_sums = [0.0] * class_count
    valid_prediction_count = 0
    for prediction in predictions:
        probs = prediction.probs
        if probs is None:
            continue
        valid_prediction_count += 1
        for idx in range(class_count):
            score_sums[idx] += float(probs.data[idx].item())

    if valid_prediction_count == 0:
        raise HTTPException(status_code=500, detail="Model did not return probabilities")

    averaged_scores = [score / valid_prediction_count for score in score_sums]
    all_scores = {names[idx]: averaged_scores[idx] for idx in range(class_count)}
    sorted_indices = sorted(range(class_count), key=lambda idx: averaged_scores[idx], reverse=True)
    top_index = sorted_indices[0]
    top_confidence = averaged_scores[top_index]
    second_confidence = averaged_scores[sorted_indices[1]] if len(sorted_indices) > 1 else 0.0
    confidence_margin = top_confidence - second_confidence
    entropy = -sum(score * math.log(score + 1e-12) for score in averaged_scores)
    is_reliable = (
        top_confidence >= MIN_TOP1_CONFIDENCE
        and confidence_margin >= MIN_TOP1_MARGIN
        and entropy <= MAX_PREDICTION_ENTROPY
        and not has_low_resolution
    )
    warning_parts: list[str] = []
    if not has_camera_metadata:
        warning_parts.append("Image has no camera metadata (possible pasted/online image)")
    if has_low_resolution:
        warning_parts.append(
            f"Image resolution is low ({width}x{height}); recommended at least {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}"
        )
    if top_confidence < MIN_TOP1_CONFIDENCE or confidence_margin < MIN_TOP1_MARGIN or entropy > MAX_PREDICTION_ENTROPY:
        warning_parts.append("Prediction confidence is low")

    message = (
        ". ".join(warning_parts) + ". Result may be less reliable."
        if warning_parts
        else "Prediction passed reliability checks."
    )

    return {
        "label": names[top_index] if is_reliable else names[top_index],
        "confidence": top_confidence,
        "all_scores": all_scores,
        "is_reliable": is_reliable,
        "message": message,
        "has_camera_metadata": has_camera_metadata,
        "has_low_resolution": has_low_resolution,
        "ensemble_views": valid_prediction_count,
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
