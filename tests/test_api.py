"""API tests that run without model weights, using a stand-in swapper."""
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_swapper
from app.swapper import DetectedFace, SwapError, SwapResult


class FakeSwapper:
    ready = True
    available_restorers = []
    masks_available = {"face_parser": False, "occluder": False}

    def load(self):
        pass

    def detect(self, img):
        return [DetectedFace(0, [10, 10, 50, 50], 0.99, "F", 30, [0.0, 5.0, 0.0])]

    def swap(self, source, target, opts):
        if opts.target_index > 0:
            raise SwapError("Target face index out of range (found 1).")
        if opts.restorer not in ("auto", "none", "codeformer", "gfpgan", "gpen_1024", "gpen_512", "restoreformer"):
            raise SwapError("bad restorer")
        out = target.copy()
        out[10:50, 10:50] = source[10:50, 10:50]
        return SwapResult(out, ["a hint"])


def png(color) -> bytes:
    img = np.full((64, 64, 3), color, dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture
def client():
    app.dependency_overrides[get_swapper] = lambda: FakeSwapper()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def files():
    return {"source": ("s.png", png(255), "image/png"), "target": ("t.png", png(0), "image/png")}


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fast Swap" in r.text


def test_health(client):
    h = client.get("/health").json()
    assert h["models_loaded"] is True and h["restorers"] == []


def test_detect(client):
    r = client.post("/api/detect", files={"image": ("a.png", png(128), "image/png")})
    assert r.status_code == 200
    face = r.json()["faces"][0]
    assert face["bbox"] == [10, 10, 50, 50] and face["pose"] == [0.0, 5.0, 0.0]


def test_swap_requires_consent(client):
    assert client.post("/api/swap", files=files()).status_code == 400


def test_swap_returns_image_and_warnings(client):
    r = client.post("/api/swap", files=files(), data={"consent": "true"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert json.loads(r.headers["x-fastswap-warnings"]) == ["a hint"]
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out[20, 20].tolist() == [255, 255, 255] and out[60, 60].tolist() == [0, 0, 0]


def test_png_is_labelled_as_edited(client):
    r = client.post("/api/swap", files=files(), data={"consent": "true"})
    assert "AI-edited" in Image.open(io.BytesIO(r.content)).info["Description"]


def test_jpeg_is_labelled_as_edited(client):
    r = client.post("/api/swap", files=files(), data={"consent": "true", "output_format": "jpg"})
    assert r.headers["content-type"] == "image/jpeg"
    assert "AI-edited" in Image.open(io.BytesIO(r.content)).getexif()[0x010E]


def test_bad_index_is_422(client):
    r = client.post("/api/swap", files=files(), data={"consent": "true", "target_face_index": "3"})
    assert r.status_code == 422


def test_bad_options_are_422(client):
    for data in ({"blend_mode": "nope"}, {"restorer": "magic"}, {"lighting": "2"}):
        r = client.post("/api/swap", files=files(), data={"consent": "true", **data})
        assert r.status_code == 422, data


def test_garbage_upload_is_400(client):
    r = client.post("/api/detect", files={"image": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400
