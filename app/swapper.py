"""Face detection + swapping pipeline.

Detection, landmarks, head pose and identity embeddings come from InsightFace
(buffalo_l). The swap uses inswapper_128. Because that model only outputs
128x128 pixels, `swap_resolution` 256/512 runs it on 4/16 sub-pixel-shifted
samples of the face and interleaves the results ("pixel boost"), which gives
a genuinely higher-resolution face. Blending and the optional helper models
live in blend.py / face_models.py.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .blend import BLEND_MODES, BlendOptions, Helpers, blend_face
from .config import Settings
from .face_models import FaceParser, Occluder, Restorer

log = logging.getLogger("fastswap")

SWAP_RESOLUTIONS = (128, 256, 512)
RESTORERS = ("none", "auto", "codeformer", "gfpgan", "gpen_1024", "gpen_512", "restoreformer")


class SwapError(Exception):
    """A problem with the user's input (no face found, bad index...)."""


@dataclass
class DetectedFace:
    index: int
    bbox: list[float]  # x1, y1, x2, y2
    score: float
    gender: str | None
    age: int | None
    pose: list[float] | None = None  # pitch, yaw, roll in degrees

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class SwapOptions:
    source_index: int = 0
    target_index: int = -1  # -1 = every face in the target
    swap_resolution: int = 256
    restorer: str = "auto"
    blend: BlendOptions = field(default_factory=BlendOptions)
    occlusion: bool = True  # use face parsing + occlusion masks when installed


@dataclass
class SwapResult:
    image: np.ndarray
    warnings: list[str]


class FaceSwapper:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._analyser = None
        self._swapper = None
        self._lock = threading.Lock()
        p, s = settings.providers, settings
        self.restorers = {
            name: Restorer(s.model_path(file), p, fidelity=s.codeformer_fidelity)
            for name, file in s.restorer_models.items()
        }
        self.parser = FaceParser(s.model_path(s.parser_model), p)
        self.occluder = Occluder(s.model_path(s.occluder_model), p)

    # ---------- model loading ----------
    def load(self) -> None:
        """Load the detector and swapper. Safe to call more than once."""
        self._load_analyser()
        self._load_swapper()

    def _load_analyser(self):
        with self._lock:
            if self._analyser is None:
                from insightface.app import FaceAnalysis

                s = self.settings
                log.info("Loading face detector '%s' with providers %s", s.detector_pack, s.providers)
                analyser = FaceAnalysis(name=s.detector_pack, root=str(s.models_dir), providers=s.providers)
                analyser.prepare(ctx_id=0, det_size=(s.det_size, s.det_size))
                self._analyser = analyser
        return self._analyser

    def _load_swapper(self):
        with self._lock:
            if self._swapper is None:
                import insightface

                path = self.settings.swapper_path
                if not path.exists():
                    raise FileNotFoundError(
                        f"Swap model not found at {path}. Run: python scripts/download_models.py"
                    )
                self._swapper = insightface.model_zoo.get_model(str(path), providers=self.settings.providers)
        return self._swapper

    @property
    def ready(self) -> bool:
        return self._analyser is not None and self._swapper is not None

    @property
    def available_restorers(self) -> list[str]:
        return [n for n, r in self.restorers.items() if r.available]

    @property
    def masks_available(self) -> dict:
        return {"face_parser": self.parser.available, "occluder": self.occluder.available}

    def _pick_restorer(self, name: str) -> Restorer | None:
        if name == "none":
            return None
        if name == "auto":
            avail = self.available_restorers
            return self.restorers[avail[0]] if avail else None
        r = self.restorers.get(name)
        if r is None or not r.available:
            raise SwapError(f"The '{name}' restorer is not installed. Run scripts/download_models.py.")
        return r

    # ---------- inference ----------
    def _faces(self, img: np.ndarray):
        faces = self._load_analyser().get(img)
        return sorted(faces, key=lambda f: f.bbox[0])  # left to right, matches the UI

    def detect(self, img: np.ndarray) -> list[DetectedFace]:
        out = []
        for i, f in enumerate(self._faces(img)):
            gender = getattr(f, "sex", None)
            age = getattr(f, "age", None)
            pose = f.get("pose")
            out.append(
                DetectedFace(
                    index=i,
                    bbox=[round(float(v), 1) for v in f.bbox],
                    score=round(float(f.det_score), 3),
                    gender=gender if isinstance(gender, str) else None,
                    age=int(age) if age is not None else None,
                    pose=None if pose is None else [round(float(v), 1) for v in pose],
                )
            )
        return out

    def _swap_aligned(self, img: np.ndarray, target_face, source_face, size: int):
        """Run inswapper at `size` px (a multiple of 128) via sub-pixel tiling."""
        from insightface.utils import face_align

        sw = self._load_swapper()
        k = size // 128
        aligned, M = face_align.norm_crop2(img, target_face.kps, size)
        latent = source_face.normed_embedding.reshape((1, -1)) @ sw.emap
        latent = (latent / np.linalg.norm(latent)).astype(np.float32)
        out = np.zeros(aligned.shape, np.float32)
        for i in range(k):
            for j in range(k):
                tile = np.ascontiguousarray(aligned[i::k, j::k])
                blob = cv2.dnn.blobFromImage(tile, 1.0 / 255, (128, 128), (0, 0, 0), swapRB=True)
                pred = sw.session.run(sw.output_names, {sw.input_names[0]: blob, sw.input_names[1]: latent})[0]
                out[i::k, j::k] = pred[0].transpose(1, 2, 0)[:, :, ::-1]
        return (out * 255).clip(0, 255).astype(np.uint8), M

    @staticmethod
    def _warnings(src, tgt_faces, size: int, numbers: list[int]) -> list[str]:
        warns = []
        src_px = min(src.bbox[2] - src.bbox[0], src.bbox[3] - src.bbox[1])
        if src_px < 90:
            warns.append(f"The source face is small ({int(src_px)} px). A larger, sharper photo gives a more faithful likeness.")
        sp = src.get("pose")
        for n, f in zip(numbers, tgt_faces):
            tp = f.get("pose")
            if sp is not None and tp is not None:
                yaw, pitch = abs(sp[1] - tp[1]), abs(sp[0] - tp[0])
                if yaw > 30 or pitch > 25:
                    warns.append(
                        f"Face {n}: head angle differs a lot from the source "
                        f"(yaw {yaw:.0f}°, pitch {pitch:.0f}°). Photos with similar angles blend better."
                    )
            face_px = max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1])
            if face_px > size * 1.1 and size < 512:
                warns.append(f"Face {n} is {int(face_px)} px wide; try swap resolution {min(512, size * 2)} for more detail.")
        return warns

    def swap(self, source: np.ndarray, target: np.ndarray, opts: SwapOptions | None = None) -> SwapResult:
        opts = opts or SwapOptions()
        if opts.blend.mode not in (*BLEND_MODES, "basic"):
            raise SwapError(f"Unknown blend mode '{opts.blend.mode}'.")
        if opts.swap_resolution not in SWAP_RESOLUTIONS:
            raise SwapError(f"Swap resolution must be one of {SWAP_RESOLUTIONS}.")
        if opts.restorer not in RESTORERS:
            raise SwapError(f"Restorer must be one of {RESTORERS}.")

        src_faces = self._faces(source)
        if not src_faces:
            raise SwapError("No face found in the source image.")
        if not 0 <= opts.source_index < len(src_faces):
            raise SwapError(f"Source face index {opts.source_index} is out of range (found {len(src_faces)}).")
        tgt_faces = self._faces(target)
        if not tgt_faces:
            raise SwapError("No face found in the target image.")
        if opts.target_index == -1:
            chosen, numbers = tgt_faces, list(range(1, len(tgt_faces) + 1))
        elif 0 <= opts.target_index < len(tgt_faces):
            chosen, numbers = [tgt_faces[opts.target_index]], [opts.target_index + 1]
        else:
            raise SwapError(f"Target face index {opts.target_index} is out of range (found {len(tgt_faces)}).")

        src = src_faces[opts.source_index]
        helpers = Helpers(
            restorer=self._pick_restorer(opts.restorer),
            parser=self.parser if opts.occlusion else None,
            occluder=self.occluder if opts.occlusion else None,
        )
        result = target.copy()
        for face in chosen:
            if opts.blend.mode == "basic":
                result = self._load_swapper().get(result, face, src, paste_back=True)
                continue
            # Always generate from the untouched target so faces don't affect each other.
            fake, M = self._swap_aligned(target, face, src, opts.swap_resolution)
            result = blend_face(result, fake, M, face, opts.blend, helpers)
        return SwapResult(result, self._warnings(src, chosen, opts.swap_resolution, numbers))


# ---------- image helpers ----------
def decode_image(data: bytes, max_side: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise SwapError("Could not read the image. Use JPEG, PNG or WebP.")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


EDIT_NOTE = "Face swapped with Fast Swap (AI-edited image)"
_XMP = (
    '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia">'
    f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{EDIT_NOTE}</rdf:li></rdf:Alt></dc:description>'
    '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


def encode_image(img: np.ndarray, fmt: str = "png") -> bytes:
    """Encode the result and label it as edited in the file's metadata
    (PNG text + XMP, or JPEG EXIF + XMP). The picture itself is untouched."""
    import io

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        exif = Image.Exif()
        exif[0x010E] = EDIT_NOTE  # ImageDescription
        exif[0x0131] = "Fast Swap"  # Software
        try:
            pil.save(buf, "JPEG", quality=95, subsampling=0, exif=exif, xmp=_XMP.encode())
        except TypeError:  # older Pillow without xmp=
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=95, subsampling=0, exif=exif)
    else:
        info = PngInfo()
        info.add_text("Software", "Fast Swap")
        info.add_text("Description", EDIT_NOTE)
        info.add_itxt("XML:com.adobe.xmp", _XMP)
        pil.save(buf, "PNG", pnginfo=info)
    return buf.getvalue()
