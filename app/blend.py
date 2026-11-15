"""Put a generated face back into the target photo so it matches its surroundings.

Steps, in order (each one optional):
  1. warp the swap model's aligned output into the photo (cubic, in a crop)
  2. face restoration (GFPGAN / CodeFormer) to recover fine detail
  3. mask: face parsing (skin, eyes, brows, nose, lips) x occlusion (hands,
     objects) x the area the model actually generated; landmark hull fallback
  4. tone: LAB mean/std matching; black-and-white targets become greyscale
  5. lighting: the target's large-scale shading (shadows, light direction) is
     transferred while the new face keeps its own fine detail
  6. sharpness: blur the new face to the target photo's level of focus
  7. grain: add noise to match the photo's sensor/film grain
  8. composite: feathered alpha or Poisson blending
Everything happens in a crop around the face, so large photos stay fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .face_models import FaceParser, Occluder, Restorer, ffhq_matrix

BLEND_MODES = ("feather", "poisson")


@dataclass
class BlendOptions:
    mode: str = "feather"
    color_match: bool = True
    lighting: float = 0.8  # 0 = off, 1 = full shading transfer
    sharpness_match: bool = True
    grain_match: bool = True
    feather: float = 1.0
    restore_strength: float = 0.8
    seed: int | None = 0


@dataclass
class Helpers:
    restorer: Restorer | None = None
    parser: FaceParser | None = None
    occluder: Occluder | None = None


@dataclass
class Region:
    """Everything about one face, in crop coordinates."""

    x0: int
    y0: int
    crop: np.ndarray  # original target pixels
    fake: np.ndarray  # generated face warped into the crop
    valid: np.ndarray  # uint8, where `fake` holds generated pixels
    kps: np.ndarray
    landmarks: np.ndarray | None
    bbox: np.ndarray
    face_size: float = field(init=False)

    def __post_init__(self):
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        self.face_size = float(np.sqrt(max(w * h, 1.0)))


# ---------------------------------------------------------------- geometry
def make_region(target: np.ndarray, fake_aligned: np.ndarray, M: np.ndarray, face) -> Region | None:
    H, W = target.shape[:2]
    size = fake_aligned.shape[0]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], np.float32) @ IM.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    pad = int(0.25 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    IM[:, 2] -= (x0, y0)
    cw, ch = x1 - x0, y1 - y0
    fake = cv2.warpAffine(fake_aligned, IM, (cw, ch), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full((size, size), 255, np.uint8), IM, (cw, ch), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8), iterations=max(2, size // 64))
    off = np.array([x0, y0], np.float32)
    lm = getattr(face, "landmark_2d_106", None)
    return Region(
        x0=x0, y0=y0, crop=target[y0:y1, x0:x1], fake=fake, valid=valid,
        kps=face.kps - off, landmarks=None if lm is None else lm - off,
        bbox=face.bbox - np.concatenate([off, off]),
    )


def _to_aligned(img: np.ndarray, A: np.ndarray, size=512) -> np.ndarray:
    return cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _from_aligned(img: np.ndarray, A: np.ndarray, shape) -> np.ndarray:
    return cv2.warpAffine(img, cv2.invertAffineTransform(A), (shape[1], shape[0]), flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- restore
def restore(r: Region, restorer: Restorer, strength: float) -> None:
    composite = np.where(r.valid[..., None] > 0, r.fake, r.crop)
    n = restorer.size
    A = ffhq_matrix(r.kps, n)
    restored = _from_aligned(restorer(_to_aligned(composite, A, n)), A, r.crop.shape)
    inside = _from_aligned(np.full((n, n), 1.0, np.float32), A, r.crop.shape)
    inside = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=4)[..., None] * strength
    r.fake = (inside * restored + (1 - inside) * r.fake).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- mask
def _hull_mask(r: Region) -> np.ndarray:
    mask = np.zeros(r.crop.shape[:2], np.uint8)
    if r.landmarks is not None:
        pts = r.landmarks.astype(np.float32)
        x, y, w, h = cv2.boundingRect(pts)
        top = pts[pts[:, 1] < y + h * 0.35].copy()
        top[:, 1] -= h * 0.10  # reach above the brows
        cv2.fillConvexPoly(mask, cv2.convexHull(np.concatenate([pts, top])).astype(np.int32), 255)
    else:
        x1, y1, x2, y2 = r.bbox
        cv2.ellipse(mask, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    (int((x2 - x1) * 0.45), int((y2 - y1) * 0.55)), 0, 0, 360, 255, -1)
    return mask.astype(np.float32) / 255


def face_mask(r: Region, helpers: Helpers) -> np.ndarray:
    """Soft-edged 0..1 mask of where the new face should show."""
    use_parser = helpers.parser is not None and helpers.parser.available
    use_occluder = helpers.occluder is not None and helpers.occluder.available
    if use_parser or use_occluder:
        A = ffhq_matrix(r.kps)
        aligned_target = _to_aligned(r.crop, A)
        m = np.ones((512, 512), np.float32)
        if use_parser:
            m *= helpers.parser(aligned_target)
        if use_occluder:
            m *= helpers.occluder(aligned_target)
        mask = _from_aligned(m, A, r.crop.shape).clip(0, 1)
        if not use_parser:
            mask *= _hull_mask(r)
    else:
        mask = _hull_mask(r)
    # Clean up small specks, then stay inside the generated area.
    mask = cv2.morphologyEx((mask > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (mask * (r.valid > 0)).astype(np.float32)


# ---------------------------------------------------------------- tone & light
def _masked_blur(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that only averages pixels inside m (normalised convolution)."""
    num = cv2.GaussianBlur(x * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def is_greyscale(img: np.ndarray, m: np.ndarray) -> bool:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = m > 0
    return sel.any() and float(lab[..., 1][sel].std() + lab[..., 2][sel].std()) < 3.0


def match_tone(r: Region, m: np.ndarray, lighting: float) -> None:
    sel = m > 0
    if sel.sum() < 64:
        return
    f = cv2.cvtColor(r.fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    t = cv2.cvtColor(r.crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    grey = is_greyscale(r.crop, m)
    for c in range(3):
        fm, fs = f[..., c][sel].mean(), f[..., c][sel].std() + 1e-6
        tm, ts = t[..., c][sel].mean(), t[..., c][sel].std() + 1e-6
        lo, hi = (0.7, 1.4) if c == 0 else (0.0, 1.5)
        f[..., c] = (f[..., c] - fm) * float(np.clip(ts / fs, lo, hi)) + tm
    if lighting > 0:
        # Replace the face's large-scale shading with the target's, keep fine detail.
        sigma = max(2.0, r.face_size * 0.08)
        mf = m.astype(np.float32)
        for c in (0,) if grey else (0, 1, 2):
            low_f = _masked_blur(f[..., c], mf, sigma)
            low_t = _masked_blur(t[..., c], mf, sigma)
            f[..., c] += lighting * (low_t - low_f)
    if grey:  # black-and-white / sepia: take the target's exact tint
        f[..., 1] = t[..., 1][sel].mean()
        f[..., 2] = t[..., 2][sel].mean()
    r.fake = cv2.cvtColor(f.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- focus & grain
def _band_energy(img: np.ndarray, sel: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = cv2.GaussianBlur(g, (0, 0), 0.8) - cv2.GaussianBlur(g, (0, 0), 2.0)
    return float(band[sel].std())


def match_sharpness(r: Region, m: np.ndarray) -> None:
    sel = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=3) > 0
    if sel.sum() < 64:
        return
    target = _band_energy(r.crop, sel)
    if _band_energy(r.fake, sel) <= target * 1.05:
        return
    best = r.fake
    for sigma in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
        cand = cv2.GaussianBlur(r.fake, (0, 0), sigma)
        best = cand
        if _band_energy(cand, sel) <= target:
            break
    r.fake = best


def _noise_level(img: np.ndarray, sel: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 1.0))[sel].std())


def match_grain(r: Region, m: np.ndarray, seed: int | None) -> None:
    sel = m > 0
    if not sel.any():
        return
    missing = _noise_level(r.crop, sel) ** 2 - _noise_level(r.fake, sel) ** 2
    if missing <= 0.25:
        return
    noise = np.random.default_rng(seed).normal(0, np.sqrt(missing), r.fake.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.5)[..., None]
    r.fake = (r.fake.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- composite
def composite(r: Region, m: np.ndarray, opts: BlendOptions) -> np.ndarray:
    hard = (m > 0).astype(np.uint8) * 255
    if opts.mode == "poisson":
        try:
            ys, xs = np.where(hard > 0)
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            border = np.zeros_like(hard)
            border[2:-2, 2:-2] = 255
            cloned = cv2.seamlessClone(r.fake, r.crop, cv2.bitwise_and(hard, border), center, cv2.NORMAL_CLONE)
            sigma = max(1.0, r.face_size * 0.015)
            core = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=max(1, int(sigma)))
            a = cv2.GaussianBlur(core, (0, 0), sigma).astype(np.float32)[..., None] / 255
            return (a * cloned + (1 - a) * r.crop).astype(np.uint8)
        except cv2.error:
            pass
    # Pull the edge in by ~1 sigma, then blur, so the fade stays inside the face.
    sigma = max(1.0, r.face_size * 0.025 * opts.feather)
    core = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=max(1, int(sigma)))
    a = cv2.GaussianBlur(core, (0, 0), sigma).astype(np.float32) / 255
    a = (a * (r.valid > 0))[..., None]
    return (a * r.fake.astype(np.float32) + (1 - a) * r.crop.astype(np.float32)).clip(0, 255).astype(np.uint8)


def blend_face(
    target: np.ndarray,
    fake_aligned: np.ndarray,
    M: np.ndarray,
    face,
    opts: BlendOptions | None = None,
    helpers: Helpers | None = None,
) -> np.ndarray:
    """Return a copy of `target` with the generated face blended in.

    `fake_aligned` is the swap model output (any size), `M` the matrix that
    aligned the target face into it.
    """
    opts = opts or BlendOptions()
    helpers = helpers or Helpers()
    r = make_region(target, fake_aligned, M, face)
    if r is None:
        return target
    if helpers.restorer is not None and helpers.restorer.available and opts.restore_strength > 0:
        restore(r, helpers.restorer, opts.restore_strength)
    m = face_mask(r, helpers)
    if m.max() == 0:
        return target
    if opts.color_match:
        match_tone(r, m, opts.lighting)
    if opts.sharpness_match:
        match_sharpness(r, m)
    if opts.grain_match:
        match_grain(r, m, opts.seed)
    out = target.copy()
    h, w = r.crop.shape[:2]
    out[r.y0:r.y0 + h, r.x0:r.x0 + w] = composite(r, m, opts)
    return out
