# Models and runtime

All neural inference in the default path is hosted. No weight downloads are included.

| Stage | Provider/model | Configuration |
|---|---|---|
| Vocal separation | fal.ai [`fal-ai/demucs`](https://fal.ai/models/fal-ai/demucs) | `model=htdemucs` (fal defaults to `htdemucs_6s`, so it is pinned), `output_format=wav`; stems vocals, drums, bass, other |
| Depth, default | fal.ai [`fal-ai/depth-anything-video`](https://fal.ai/models/fal-ai/depth-anything-video) | Video Depth Anything; `model=VDA-Large`, `colormap=inferno` (override with `FAL_DEPTH_MODEL` / `FAL_DEPTH_COLORMAP`), `resolution=auto`; the returned colors are preserved exactly |
| Depth alternative | [Video-Depth-Anything Space](https://huggingface.co/spaces/depth-anything/Video-Depth-Anything) | public GPU quota/availability; optional HF_TOKEN |
| Segmentation | fal.ai [`fal-ai/sam-3/video-rle`](https://fal.ai/models/fal-ai/sam-3/video-rle) | `video_url`, `prompt`, `apply_mask=false`, `detection_threshold=0.5`; per-frame RLE masks are decoded locally (`rle_masks.py`) into `mask_NNNNN.png` |
| Generation | Enhancor Seedance 2.5 | Edit or multi_reference; draft then approved 1080p |
| Pitch | Rubber Band CLI | +3 semitones, time ratio 1 |
| Video/audio IO | PyAV / FFmpeg libraries | normalization, compositing, AAC input vocals, original audio packet remux |

Input and output field names for the fal.ai endpoints were taken from fal's published model pages on 2026-11-01. fal documents the `video-rle` output only as per-frame RLE masks, without stating the encoding. `rle_masks.py` therefore accepts COCO-style RLE (compressed string, uncompressed counts, or a `{size, counts}` dict, optionally per object) and refuses to guess when the entry count does not match the frame count. Run `fal_check.py` once on a very short clip to confirm your account's response decodes. If a model becomes unavailable, inspect its current schema before changing the adapter. [fal.ai model API docs](https://fal.ai/models).

The [SAM3-Video Hugging Face Space](https://huggingface.co/spaces/linoyts/SAM3-Video) was an earlier exploration; it is not the active raw-mask implementation. Never describe a green visualization overlay as a raw mask. Public Spaces may change their Gradio API. An RGB segmentation retry is a deliberate repair, not a different depth color treatment.

`requirements.txt` records versions from the development environment. Setup installs them into a clean venv; the package's local tests are also runnable using a compatible existing environment. Hosted availability, price and access are account-dependent; no free-inference claim is made. Local self-hosting of model weights is not configured here.
