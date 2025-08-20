# Services Caching Guide

This document explains how module-level model caching and dataset/result caching work in the backend services. It also clarifies why static and dynamic metadata are separated and the performance optimizations involved.

## Overview

- Static dataset metadata: built from files and optional CSVs, cached in Redis per dataset.
- Dynamic predicted metadata: model inference results keyed by model and file hash, cached in Redis.
- Model objects (Whisper ASR, wav2vec2 emotion): cached in-process at module scope for fast reuse.

---

## Module-Level Model Caching (`app/services/model_loader_service.py`)

- Device/dtype:

  - `_device` selects CUDA if available, else CPU.
  - `_torch_dtype` is float16 when CUDA is available, else float32.

- Whisper (ASR) pipelines:

  - Globals: `_asr_pipelines: dict[str, pipeline]`, `_asr_locks: dict[str, threading.Lock]`.
  - `get_asr_pipeline(model_id: str)` lazily creates and caches a pipeline per `model_id`.
  - Double-checked locking prevents duplicate loads under concurrency.
  - Pipelines live for the life of the Python process (per-worker); they are not stored in Redis.

- Wav2Vec2 (emotion) components:

  - Globals: `_emotion_feature_extractor`, `_emotion_model`, `_emotion_lock`.
  - `get_emotion_components()` lazily loads and caches both. `eval()` is set on the model.

- Persistence across user model changes:

  - Users switching models (e.g., `openai/whisper-base` ↔ `openai/whisper-large-v3`) triggers `get_asr_pipeline(model_id)`; if already cached, reused instantly.
  - If a model is already cached, it’s reused instantly. The first use of a given model may incur a one‑time load/warm‑up  delay (and, on CUDA, kernel/VRAM initialization). Subsequent uses are near‑instant. If a pipeline is being created concurrently, the initial call will block on the per‑model lock.
  - Multiple models may be cached simultaneously; nothing is unloaded automatically.
  - To “reset” models, restart the process or add a reset routine (not implemented).
  - Model weights are persisted on disk by Hugging Face’s cache (configure `HF_HOME` for durable storage); the in-memory objects persist only for the process lifetime.

- Optional warmup (preload):
  - `app/main.py` has a `@app.on_event("startup")` hook honoring `PRELOAD_MODELS` (comma-separated), e.g. `whisper-base,whisper-large,wav2vec2`.
  - This pre-populates the module caches to eliminate first-request latency.

---

## Static Dataset Metadata Caching (`app/services/dataset_manifest.py`)

- Public API:

  - `compute_dir_version(base: Path) -> str`: content hash of all `*.wav`, `*.mp3`, and `*_metadata.csv` (relpath, size, mtime).
  - `get_or_build_manifest(ds_id: str, base: Path, ttl: int|None = None, force: bool = False)` → `(entries, summary)`.

- Manifest build (`_scan_dataset`):

  - Scans WAV/MP3 files, reads WAV headers (stdlib `wave`) for `(duration, sample_rate)`.
  - Derives `label` from filename (RAVDESS pattern) or CSV fallback.
  - Computes stable `h` by hashing `(relpath, size, mtime)`; used to join with results.
  - Attaches per-file `meta` from `*_metadata.csv`, pruned by `meta_constants` and without duplicate `filename`.
  - Entry shape (subset): `{id, relpath, filename, size, duration, sample_rate?, label, h, meta?}`.
  - Summary includes `{total, total_bytes, label_counts?, meta_constants?}`.

- Redis keys (see `app/core/redis.py`):

  - `dataset:{id}:manifest`, `dataset:{id}:summary`, `dataset:{id}:version`.
  - TTL uses `settings.DATASET_CACHE_TTL_SECONDS`.

- Validation & invalidation:

  - Current `compute_dir_version(base)` is compared to cached `version`.
  - If unchanged and cached JSON exists, return cached values; otherwise rebuild and set keys with TTL.
  - Force rebuild via API `POST /datasets/reindex {"id": "..."}`.

- Serving to frontend (see `app/api/routes/datasets.py`):
  - `GET /datasets/files` windows/paginates the manifest and returns `sample_rate` when present.
  - `GET /datasets/summary` returns the cached summary, including `meta_constants` when available.
  - `GET /datasets/file` streams the requested file with appropriate media type.

---

## Dynamic Predicted Metadata Caching (`app/api/routes/results.py` and `app/core/redis.py`)

- Result cache API:

  - `GET /results/{model}/{h}` → `{cached: bool, payload: object|null}`.
  - `POST /results/{model}/{h}` caches a JSON payload for that `{model, h}` pair (TTL default 6h).
  - `POST /results/{model}/batch` with `{hashes: [h1, ...]}` returns a `{h -> payload|null}` map in one round trip.

- Redis keying and TTL:

  - Key: `result:{model}:{h}` (see `k_result`).
  - Stored via `cache_result(model, h, payload, ttl=6*60*60)`; retrieved via `get_result(model, h)`.

- Payload examples:
  - Whisper ASR: `{"text": "...", "segments": [...], "model_id": "openai/whisper-base", ...}` (shape depends on pipeline output).
  - Wav2Vec2 emotion: `{"label": "happy", "confidence": 0.87, "probs": {"happy": ..., ...}, "model_id": "r-f/wav2vec-english-speech-emotion-recognition"}`.

---

## Why Separate Static vs Dynamic Metadata?

- Change cadence:

  - Static metadata (filenames, sizes, durations, constants) changes only when the dataset changes.
  - Dynamic metadata (model predictions) changes per model, per parameter set, per content update.

- Invalidation strategy:

  - Static: versioned by directory snapshot; refreshed only when necessary or when forced.
  - Dynamic: short TTL, keyed by `model` and `h` to isolate results across models and data revisions.

- Payload size and reuse:

  - Static manifests can be large; caching avoids repeated filesystem scans and WAV header reads.
  - Dynamic results are small but numerous; caching avoids repeated model inference and supports batch fetches.

- UI concerns:
  - Static `meta_constants` are shown once (summary) to declutter per-row `meta`.
  - Per-row `meta` hides duplicated fields (e.g., constant columns, duplicate `filename`).

---

## Operational Notes and Optimizations

- Performance & latency:

  - Module-level model caches eliminate expensive cold starts when users switch models.
  - Directory versioning prevents unnecessary re-scans; WAV header parsing is offloaded with `asyncio.to_thread`.
  - Batch result fetch minimizes API round trips for tables.

- Memory & concurrency:

  - Double-checked locks prevent duplicate model loads.
  - Each Uvicorn worker/process maintains its own in-memory caches (models). Scale workers mindful of RAM/VRAM.

- Durability:

  - In-memory model objects are not persisted across process restarts.
  - Model weights persist on disk via the HF cache; set `HF_HOME` (and mount as a volume in containers) for durability.

- Reindexing:
  - Use `POST /datasets/reindex` with `{ "id": "ravdess_subset" | "common_voice_en_dev" }` to force a rebuild.

---

## Quick References

- Static cache code: `app/services/dataset_manifest.py`, keys in `app/core/redis.py`.
- Dynamic cache code: `app/api/routes/results.py`, helpers in `app/core/redis.py`.
- Model caches: `app/services/model_loader_service.py`, warmup in `app/main.py` via `PRELOAD_MODELS`.
- Settings: `app/core/settings.py` (`DATASET_CACHE_TTL_SECONDS`, `REDIS_URL`, etc.).
