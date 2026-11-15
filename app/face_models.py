"""Optional ONNX helper models that improve precision. None of them need PyTorch.

* Restorer  - CodeFormer, GFPGAN v1.4, GPEN-BFR (512/1024) or RestoreFormer++:
              adds back fine detail (pores, eyelashes) the swap model can't produce.
* FaceParser - BiSeNet: labels skin, eyes, brows, nose, lips, hair, glasses...
              so the mask covers exactly the face and nothing in front of it.
* Occluder  - XSeg: finds hands, microphones, etc. covering the face.

All of them work on a 512x512 crop aligned to the FFHQ face template.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("fastswap")

# Five-point FFHQ template (eyes, nose, mouth corners), as a fraction of crop size.
FFHQ_5PT = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    np.float32,
)

# BiSeNet (CelebAMask-HQ) classes that belong to the swappable face.
PARSER_FACE_CLASSES = {
    "skin": 1, "left_brow": 2, "right_brow": 3, "left_eye": 4, "right_eye": 5,
    "nose": 10, "mouth": 11, "upper_lip": 12, "lower_lip": 13,
}  # glasses (6), ears, hair, hat, neck, hands (background) are kept from the target


def ffhq_matrix(kps: np.ndarray, size: int = 512) -> np.ndarray:
    """Similarity transform mapping image coordinates -> aligned FFHQ crop."""
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), FFHQ_5PT * size, method=cv2.LMEDS)
    return M


class _OnnxModel:
    def __init__(self, path: Path, providers: list[str]):
        self.path = path
        self.providers = providers
        self._session = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.path.exists()

    @property
    def session(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                log.info("Loading %s", self.path.name)
                self._session = ort.InferenceSession(str(self.path), providers=self.providers)
        return self._session


class Restorer(_OnnxModel):
    """GFPGAN / CodeFormer. `fidelity` only applies to CodeFormer (1 = closest to input)."""

    def __init__(self, path: Path, providers: list[str], fidelity: float = 0.7):
        super().__init__(path, providers)
        self.fidelity = fidelity

    @property
    def size(self) -> int:
        """Input resolution the model expects (512 for most, 1024 for GPEN-1024)."""
        return int(self.session.get_inputs()[0].shape[2])

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        x = aligned_bgr[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        feeds = {"input": x}
        if any(i.name == "weight" for i in self.session.get_inputs()):
            feeds["weight"] = np.array(self.fidelity, dtype=np.float64)
        y = self.session.run(None, feeds)[0][0]
        y = (np.clip(y, -1, 1) + 1) / 2
        return (y.transpose(1, 2, 0)[:, :, ::-1] * 255).round().astype(np.uint8)


class FaceParser(_OnnxModel):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Returns a 0..1 float mask of the face parts in the aligned crop."""
        x = cv2.resize(aligned_bgr, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        labels = self.session.run(None, {"input": x})[0][0].argmax(0)
        mask = np.isin(labels, list(PARSER_FACE_CLASSES.values())).astype(np.float32)
        return cv2.resize(mask, aligned_bgr.shape[1::-1])


class Occluder(_OnnxModel):
    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Returns a 0..1 float mask: 1 where the face is visible (not covered)."""
        x = cv2.resize(aligned_bgr, (256, 256)).astype(np.float32)[None] / 255.0
        m = self.session.run(None, {"input": x})[0][0][:, :, 0].clip(0, 1)
        m = cv2.resize(m, aligned_bgr.shape[1::-1])
        return (cv2.GaussianBlur(m, (0, 0), 5).clip(0.5, 1) - 0.5) * 2
