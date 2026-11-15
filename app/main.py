"""FastAPI entry point: `uvicorn app.main:app --reload`."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .blend import BlendOptions
from .swapper import FaceSwapper, SwapError, SwapOptions, decode_image, encode_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fastswap")

STATIC = Path(__file__).parent / "static"
_swapper = FaceSwapper(settings)


def get_swapper() -> FaceSwapper:
    return _swapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    swapper = app.dependency_overrides.get(get_swapper, get_swapper)()
    try:
        await run_in_threadpool(swapper.load)
    except Exception as exc:  # keep the server up so /health can explain what is missing
        log.error("Models not loaded: %s", exc)
    yield


app = FastAPI(title="Fast Swap", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


async def _read(upload: UploadFile, label: str) -> bytes:
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await upload.read(limit + 1)
    if not data:
        raise HTTPException(400, f"The {label} image is empty.")
    if len(data) > limit:
        raise HTTPException(413, f"The {label} image is larger than {settings.max_upload_mb:g} MB.")
    return data


def _decode(data: bytes, label: str):
    try:
        return decode_image(data, settings.max_side)
    except SwapError as exc:
        raise HTTPException(400, f"{label.capitalize()}: {exc}") from exc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health(swapper: FaceSwapper = Depends(get_swapper)):
    return {
        "status": "ok" if swapper.ready else "models_missing",
        "models_loaded": swapper.ready,
        "restorers": swapper.available_restorers,
        "masks": swapper.masks_available,
        "providers": settings.providers,
    }


@app.post("/api/detect")
async def detect(image: UploadFile = File(...), swapper: FaceSwapper = Depends(get_swapper)):
    """List the faces in an image, ordered left to right."""
    img = _decode(await _read(image, "uploaded"), "image")
    try:
        faces = await run_in_threadpool(swapper.detect, img)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"width": img.shape[1], "height": img.shape[0], "faces": [f.as_dict() for f in faces]}


@app.post(
    "/api/swap",
    responses={200: {"content": {"image/png": {}, "image/jpeg": {}},
                     "headers": {"X-FastSwap-Warnings": {"description": "JSON list of quality hints"}}}},
    response_class=Response,
)
async def swap(
    source: UploadFile = File(..., description="Image containing the face to use"),
    target: UploadFile = File(..., description="Image whose face(s) will be replaced"),
    source_face_index: int = Form(0, description="Which face in the source to use (left to right)"),
    target_face_index: int = Form(-1, description="Which face in the target to replace; -1 for all"),
    swap_resolution: int = Form(256, description="128, 256 or 512. Higher = sharper, slower"),
    restorer: str = Form("auto", pattern="^(none|auto|codeformer|gfpgan|gpen_1024|gpen_512|restoreformer)$",
                         description="Face restoration model; auto picks the first installed of CodeFormer, GFPGAN, GPEN, RestoreFormer++"),
    restore_strength: float = Form(0.8, ge=0, le=1, description="How much of the restored face to use"),
    blend_mode: str = Form("feather", pattern="^(feather|poisson|basic)$",
                           description="feather (default), poisson, or basic (inswapper's own paste)"),
    color_match: bool = Form(True, description="Match skin tone to the target"),
    lighting: float = Form(0.8, ge=0, le=1, description="Transfer the target's shading and shadows (0 = off)"),
    occlusion: bool = Form(True, description="Keep hair, glasses and hands in front of the face"),
    sharpness_match: bool = Form(True, description="Match the target photo's focus"),
    grain_match: bool = Form(True, description="Match the photo's grain/noise"),
    output_format: str = Form("png", pattern="^(png|jpg|jpeg)$"),
    consent: bool = Form(False, description="Confirms everyone pictured agreed to this use"),
    swapper: FaceSwapper = Depends(get_swapper),
):
    if not consent:
        raise HTTPException(400, "Please confirm you have consent from the people in both images.")
    src = _decode(await _read(source, "source"), "source")
    tgt = _decode(await _read(target, "target"), "target")
    opts = SwapOptions(
        source_index=source_face_index,
        target_index=target_face_index,
        swap_resolution=swap_resolution,
        restorer=restorer,
        occlusion=occlusion,
        blend=BlendOptions(
            mode=blend_mode,
            color_match=color_match,
            lighting=lighting if color_match else 0.0,
            sharpness_match=sharpness_match,
            grain_match=grain_match,
            restore_strength=restore_strength,
        ),
    )
    try:
        result = await run_in_threadpool(swapper.swap, src, tgt, opts)
    except SwapError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    media = "image/jpeg" if output_format in ("jpg", "jpeg") else "image/png"
    return Response(
        encode_image(result.image, output_format),
        media_type=media,
        headers={"X-FastSwap-Warnings": json.dumps(result.warnings, ensure_ascii=True)},
    )
