## SoraWatermarkCleaner – Understanding

This repo removes the Sora2 watermark from videos using a YOLO-based detector plus an inpainting cleaner (LAMA via IOPaint). It also provides a Streamlit UI and a FastAPI server for batch/API usage.

### High-level architecture

- **Pipeline (`sorawm/core.py`)**: `SoraWM` orchestrates the end-to-end flow.

  - Loads a `VideoLoader` to stream frames via ffmpeg.
  - Uses `SoraWaterMarkDetector` (YOLO) to detect watermark bbox per frame.
  - Fills occasional missed detections via neighbor propagation.
  - Builds a binary mask from bbox and calls `WaterMarkCleaner` to inpaint.
  - Writes cleaned frames to a temp video via ffmpeg, then merges original audio.

- **Detector (`sorawm/watermark_detector.py`)**: YOLO model loaded from `resources/best.pt` (auto-downloaded if missing). Returns bbox, confidence, and center. Device picked via `utils/devices_utils.get_device`.

- **Cleaner (`sorawm/watermark_cleaner.py`)**: Wraps IOPaint LAMA. Ensures model exists (`cli_download_model`), initializes `ModelManager`, then performs inpainting with an `InpaintRequest`.

- **Video IO (`sorawm/utils/video_utils.py`)**: `VideoLoader` probes video metadata and streams raw BGR frames from ffmpeg stdout. Also exposes width/height/fps/total_frames/bitrate.

- **Configs (`sorawm/configs.py`)**: Central paths: `RESOURCES_DIR`, YOLO weights path, output/working/logs/data/sqlite locations, and default inpaint model name (`lama`). Creates directories on import.

- **Downloads (`sorawm/utils/download_utils.py`)**: Fetches YOLO `best.pt` from GitHub release with progress and error handling.

- **Misc watermark utils (`sorawm/utils/watermark_utls.py`)**: Template-matching helpers (not used by main pipeline; likely earlier experiments).

### Server (FastAPI)

- Entry (`start_server.py`): Initializes logging, builds app via `sorawm.server.app.init_app`, and runs uvicorn.
- App (`sorawm/server/app.py`): Creates `FastAPI` with lifespan and attaches router.
- Lifespan (`sorawm/server/lifespan.py`): On startup, initializes SQLite schema and `WMRemoveTaskWorker`; starts background worker loop.
- DB (`sorawm/server/db.py`): Async SQLAlchemy setup (SQLite + aiosqlite), session context, `init_db` for metadata create.
- Model (`sorawm/server/models.py`): `Task` table with id, paths, status, percentage, download URL, timestamps.
- Schemas (`sorawm/server/schemas.py`): `Status` enum and `WMRemoveResults` response model.
- Router (`sorawm/server/router.py`):
  - `POST /submit_remove_task`: saves uploaded video asynchronously, enqueues task, returns `task_id`.
  - `GET /get_results?remove_task_id=...`: progress/status and (when finished) a download URL.
  - `GET /download/{task_id}`: returns cleaned video file.
- Worker (`sorawm/server/worker.py`):
  - Holds an asyncio `Queue`, a singleton `SoraWM`, and upload/output dirs.
  - `create_task` inserts an UPLOADING record; `queue_task` updates DB and enqueues.
  - `run` loop processes tasks: builds output path, updates progress via callback from `SoraWM.run`, finalizes DB record, and exposes a download URL.

### UI (Streamlit)

- `app.py`: Interactive UI. Loads `SoraWM` once into session, uploads a video, runs `sora_wm.run` with a progress callback (stages 10–95%), previews and allows download of cleaned video.

### CLI example

- `example.py`: Minimal usage of `SoraWM` on `resources/dog_vs_sam.mp4` to `outputs/sora_watermark_removed.mp4`.

### Training and datasets

- `datasets/make_yolo_images.py`: Extracts frames from videos under `ROOT/videos` into `datasets/images` for labelling/training.
- `train/coco8.yaml`: YOLO dataset config (single class: `watermark`).
- `train/train.py`: Example YOLO11s training script against the dataset, evaluation and sample inference.
- `train/eval.py`: Quick load of trained weights and image prediction sanity-check.

### Data flow details (`SoraWM.run`)

1. Probe input video: width/height/fps/total_frames/bitrate.
2. Start ffmpeg writer to a temp mp4 (x264). If bitrate known, set ~1.2x for quality; else use CRF 18.
3. Iterate frames:
   - YOLO detect bbox; collect missed-frame indices; periodically emit 10–50% progress.
4. For missed indices, propagate bbox from neighbors (previous or next).
5. Iterate frames again:
   - If bbox present, build mask and inpaint; else pass-through. Write each frame to ffmpeg stdin; emit 50–95% progress.
6. Close writer, then merge original audio with cleaned temp video into final output; cleanup temp; emit up to 99%.

### Dependencies (pyproject highlights)

- FastAPI, Uvicorn, SQLAlchemy (async) for server.
- Streamlit for UI.
- Ultralytics YOLO, torch/torchvision for detection.
- IOPaint (vendored under `sorawm/iopaint`) with diffusers/transformers/einops/omegaconf.
- ffmpeg-python for video IO and muxing.

### Notable implementation choices

- Frame-level inpainting confined to YOLO bbox for speed and quality.
- Simple temporal smoothing by propagating nearest bbox to missed frames.
- Uses rawvideo pipe to avoid decoding/re-encoding overhead per frame read/write.
- Preserves original audio by remuxing after video inpainting.

### How to run

- Demo: `python example.py`.
- Streamlit UI: `streamlit run app.py`.
- Server: `python start_server.py`, then see docs at http://localhost:5344/docs.
- Environment: install via `uv sync` (Python >= 3.12, ffmpeg installed). Weights auto-download on first run.

### Limitations and notes

- Single-class detector assumes a consistent Sora watermark style; custom videos may require retraining.
- Temporal coherence relies on neighbor propagation; complex motion could benefit from tracking.
- Inpainting uses the LAMA default; alternative models in IOPaint could be configured if needed.

### Repository pointers

- Core entry: `sorawm/core.py` (`SoraWM`).
- Detector: `sorawm/watermark_detector.py` (YOLO + auto download).
- Cleaner: `sorawm/watermark_cleaner.py` (IOPaint LAMA).
- Server: `sorawm/server/*` (FastAPI + worker + SQLite).
- UI: `app.py` (Streamlit single-page app).
- Training: `datasets/`, `train/` utilities and configs.



