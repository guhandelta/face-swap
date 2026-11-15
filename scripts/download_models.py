"""Download the model weights Fast Swap needs into ./models.

    python scripts/download_models.py              # everything (~1.7 GB)
    python scripts/download_models.py --core-only  # swap model + detector only
    python scripts/download_models.py --extras     # also GPEN 512/1024 + RestoreFormer++

Models already in ./models are skipped. Override the swap model URL with
FASTSWAP_SWAPPER_URL. Several of these weights are NON-COMMERCIAL only (see README).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

ASSETS ="https://github.com/facefusion/facefusion-assets/releases/download"
URLS = {
    # name in settings -> (url, what it's for)
    "swapper": (os.getenv("FASTSWAP_SWAPPER_URL", f"{ASSETS}/models-3.0.0/inswapper_128.onnx"), "face swap"),
    "codeformer": (f"{ASSETS}/models-3.0.0/codeformer.onnx", "detail restore (CodeFormer)"),
    "gfpgan": (f"{ASSETS}/models-3.0.0/gfpgan_1.4.onnx", "detail restore (GFPGAN)"),
    "gpen_1024": (f"{ASSETS}/models-3.0.0/gpen_bfr_1024.onnx", "detail restore (GPEN 1024, extra)"),
    "gpen_512": (f"{ASSETS}/models-3.0.0/gpen_bfr_512.onnx", "detail restore (GPEN 512, extra)"),
    "restoreformer": (f"{ASSETS}/models-3.0.0/restoreformer_plus_plus.onnx", "detail restore (RestoreFormer++, extra)"),
    "parser": (f"{ASSETS}/models-3.0.0/bisenet_resnet_34.onnx", "face parsing mask"),
    "occluder": (f"{ASSETS}/models-3.1.0/xseg_1.onnx", "occlusion mask"),
}


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"✓ {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"↓ {url}")
    sha = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            sha.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    tmp.rename(dest)
    print(f"\n✓ saved {dest} (sha256 {sha.hexdigest()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-only", action="store_true",
                    help="only the swap model and detector (skip restore/mask models, ~0.9 GB)")
    ap.add_argument("--extras", action="store_true",
                    help="also fetch GPEN 512/1024 and RestoreFormer++ (~860 MB more)")
    args = ap.parse_args()
    extras = {"gpen_1024", "gpen_512", "restoreformer"}

    files = {
        "swapper": settings.swapper_path,
        "codeformer": settings.model_path(settings.restorer_models["codeformer"]),
        **{k: settings.model_path(settings.restorer_models[k]) for k in ("gfpgan", *sorted(extras))},
        "parser": settings.model_path(settings.parser_model),
        "occluder": settings.model_path(settings.occluder_model),
    }
    for key, dest in files.items():
        if args.core_only and key != "swapper":
            continue
        if key in extras and not args.extras:
            continue
        url, purpose = URLS[key]
        print(f"[{purpose}]")
        fetch(url, dest)

    # InsightFace downloads its detector pack on first use; trigger it now.
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        sys.exit(f"insightface is not installed in this Python ({sys.executable}).\n"
                 "Activate your virtualenv and run: python -m pip install -r requirements.txt")

    FaceAnalysis(name=settings.detector_pack, root=str(settings.models_dir), providers=["CPUExecutionProvider"])
    print(f"✓ detector pack '{settings.detector_pack}' ready")


if __name__ == "__main__":
    main()
