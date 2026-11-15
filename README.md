# Fast Swap

A FastAPI app that puts the face from one photo onto the faces in another, with a small drag-and-drop web UI.

**Pipeline** (each step can be switched off):

1. **Detect** faces, 106 landmarks and head pose (InsightFace `buffalo_l`).
2. **Swap** with `inswapper_128`. At 256/512 px "quality" the model runs on 4/16 sub-pixel-shifted samples of the face and the outputs are interleaved, giving a genuinely higher-resolution face instead of an upscaled 128 px one.
3. **Restore detail** with CodeFormer, GFPGAN, GPEN or RestoreFormer++ (ONNX, no PyTorch).
4. **Mask** with face parsing (BiSeNet) and occlusion detection (XSeg), so hair, glasses, ears, hands and objects in front of the face stay from the original photo.
5. **Match** the original photo: skin tone (LAB statistics), lighting and shadows (the target's large-scale shading is transferred while the new face keeps its fine detail), focus (blurred to the photo's sharpness), and grain. Black-and-white and sepia photos get a matching monochrome face.
6. **Blend** with a feathered edge or Poisson blending.
7. **Label** the output file as AI-edited in its metadata (PNG text/XMP or JPEG EXIF/XMP, using the IPTC "composite with trained algorithmic media" tag). The image pixels aren't marked.

The API also returns quality hints, for example when the source and target head angles differ a lot.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py          # ~1.7 GB; --core-only for just the swap model, --extras for more restorers
uvicorn app.main:app --reload
```

Open http://localhost:8000, drop a source photo and a target photo, click a face box to choose which face to use or replace, tick the consent box and press **Swap faces**. API docs are at `/docs`.

### GPU
Replace `onnxruntime` with `onnxruntime-gpu` (NVIDIA, CUDA 12). The app picks CUDA → CoreML → CPU automatically; force a choice with `FASTSWAP_PROVIDERS=CPUExecutionProvider`. On CPU a swap takes roughly 1–3 s per face.

### Speed (CPU, one face)
Measured on a cloud CPU, including face detection on both photos: basic ≈ 3 s, 256 px with everything on ≈ 7 to 8 s, 512 px ≈ 13 s. An NVIDIA GPU or Apple Silicon (CoreML) is much faster.

### Using it from an Android app
Keep the models on a server running this API and have the app upload the two photos to `/api/swap`. The full model set (~1.7 GB) is too large to bundle in an app.

### Docker
```bash
docker build -t fast-swap .
docker run -p 8000:8000 -v "$PWD/models:/srv/models" fast-swap
```

## API

| Method | Path | Body (multipart) | Returns |
|---|---|---|---|
| GET | `/health` | | model status, providers |
| POST | `/api/detect` | `image` | faces left-to-right: `index`, `bbox`, `score`, `gender`, `age` |
| POST | `/api/swap` | `source`, `target`, `consent=true`, optional `source_face_index` (0), `target_face_index` (-1 = all), `swap_resolution` (128/**256**/512), `restorer` (**auto**/codeformer/gfpgan/gpen_1024/gpen_512/restoreformer/none), `restore_strength` (0.8), `blend_mode` (**feather**/poisson/basic), `color_match`, `lighting` (0 to 1, 0.8), `occlusion`, `sharpness_match`, `grain_match`, `output_format` (png/jpg) | the swapped image; quality hints in the `X-FastSwap-Warnings` header (JSON list) |

```bash
curl -F source=@me.jpg -F target=@group.jpg -F consent=true -F target_face_index=2 \
     http://localhost:8000/api/swap -o out.png
```

Errors: 400 bad upload or missing consent, 413 too large, 422 no face / bad index, 503 models missing.

## Configuration (environment variables)

| Variable | Default | |
|---|---|---|
| `FASTSWAP_MODELS_DIR` | `./models` | where weights live |
| `FASTSWAP_PROVIDERS` | auto | comma-separated ONNX Runtime providers |
| `FASTSWAP_DET_SIZE` | `640` | detector input size; raise for small faces |
| `FASTSWAP_MAX_UPLOAD_MB` | `10` | per-file upload limit |
| `FASTSWAP_MAX_SIDE` | `2048` | larger images are downscaled first |
| `FASTSWAP_CODEFORMER_FIDELITY` | `0.7` | 0 = most detail added, 1 = closest to the swap output |
| `FASTSWAP_SWAPPER_URL` | see script | alternative download location for the swap model |

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
The tests swap in a fake model, so they run without any weights.

## Model licensing (read before using commercially)
The download script fetches all weights from the FaceFusion assets mirror on GitHub. Their licences are separate from this project's code:

| Model | Used for | Licence |
|---|---|---|
| inswapper_128, buffalo_l (InsightFace) | swap, detection | **non-commercial research only**; InsightFace sells commercial licences |
| CodeFormer | detail restore | **S-Lab License 1.0, non-commercial** |
| GFPGAN v1.4 | detail restore | Apache-2.0 |
| GPEN-BFR 512/1024 (`--extras`) | detail restore | research release; check the GPEN repo's terms before commercial use |
| RestoreFormer++ (`--extras`) | detail restore | check the RestoreFormer repo's terms before commercial use |
| BiSeNet face parsing | mask | MIT |
| XSeg (DeepFaceLab) | occlusion mask | GPL-3.0 |

A commercial app (including a paid Android app) needs an InsightFace commercial licence or a different swap model, and GFPGAN instead of CodeFormer.

## Responsible use
Face swapping can hurt real people. By running this software you agree to:
- Only use photos of people who have **given consent**, and only for the purpose they agreed to.
- Never create content that **impersonates** someone, deceives viewers, harasses, or is sexual in nature without explicit consent; non-consensual intimate imagery is illegal in many places.
- **Label** swapped media as altered when you share it.
- Follow the laws where you live (likeness, privacy and deepfake laws vary).

The API refuses to swap unless `consent=true` is sent, and every output file carries an "AI-edited" metadata label. Please keep that label when you build on this. Images are processed in memory and never written to disk.
