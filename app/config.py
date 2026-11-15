"""Runtime settings, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _providers() -> list[str]:
    raw = os.getenv("FASTSWAP_PROVIDERS", "")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:  # pragma: no cover - onnxruntime missing
        available = []
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("FASTSWAP_MODELS_DIR", ROOT / "models")))
    swapper_model: str = os.getenv("FASTSWAP_SWAPPER_MODEL", "inswapper_128.onnx")
    detector_pack: str = os.getenv("FASTSWAP_DETECTOR_PACK", "buffalo_l")
    restorer_models: dict = field(default_factory=lambda: {
        "codeformer": os.getenv("FASTSWAP_CODEFORMER_MODEL", "codeformer.onnx"),
        "gfpgan": os.getenv("FASTSWAP_GFPGAN_MODEL", "gfpgan_1.4.onnx"),
        "gpen_1024": os.getenv("FASTSWAP_GPEN1024_MODEL", "gpen_bfr_1024.onnx"),
        "gpen_512": os.getenv("FASTSWAP_GPEN512_MODEL", "gpen_bfr_512.onnx"),
        "restoreformer": os.getenv("FASTSWAP_RESTOREFORMER_MODEL", "restoreformer_plus_plus.onnx"),
    })
    parser_model: str = os.getenv("FASTSWAP_PARSER_MODEL", "bisenet_resnet_34.onnx")
    occluder_model: str = os.getenv("FASTSWAP_OCCLUDER_MODEL", "xseg_1.onnx")
    codeformer_fidelity: float = float(os.getenv("FASTSWAP_CODEFORMER_FIDELITY", "0.7"))
    det_size: int = int(os.getenv("FASTSWAP_DET_SIZE", "640"))
    max_upload_mb: float = float(os.getenv("FASTSWAP_MAX_UPLOAD_MB", "10"))
    max_side: int = int(os.getenv("FASTSWAP_MAX_SIDE", "2048"))
    providers: list[str] = field(default_factory=_providers)

    @property
    def swapper_path(self) -> Path:
        return self.models_dir / self.swapper_model

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


settings = Settings()
