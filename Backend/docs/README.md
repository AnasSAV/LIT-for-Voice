# Cache Internals: Dataset Manifests and Results

This document explains how static dataset metadata and model results are cached in Redis, how keys are constructed, and what the schemas look like.

## Overview

- Dataset files are scanned from `Backend/data/` under one of:
  - `dev/ravdess_subset/`
  - `raw/ravdess_full/`
  - `raw/common_voice_en/`
- The scan builds a manifest and a summary, both cached in Redis per dataset id.
- Model results are cached separately by `{model}` and file hash `{h}`.

## Redis Keys

- `dataset:{id}:manifest`: JSON array of file entries
- `dataset:{id}:summary`: JSON object of dataset-level stats
- `dataset:{id}:version`: string version; if this changes, manifest/summary are rebuilt
- `result:{model}:{h}`: JSON payload for model `{model}` keyed by file hash `{h}`

## Versioning and Invalidation

`compute_dir_version(base)` hashes over:
- Every `*.wav` `relpath`, `size`, `mtime`
- Every `*_metadata.csv` `relpath`, `size`, `mtime`

If any of the above changes, the version changes and the cache is rebuilt on the next request.

## Entry Schema (dataset manifest)

Produced by `Backend/app/services/dataset_manifest.py` `_scan_dataset()` and returned by `GET /datasets/files`:

```json
{
  "id": "03-01-01-01-01-01-03.wav",
  "filename": "03-01-01-01-01-01-03.wav",
  "relpath": "03-01-01-01-01-01-03.wav",
  "size": 379588,
  "duration": 3.43677083333333,
  "sample_rate": 16000,
  "label": "neutral",
  "h": "f7fb16169ec10e42c6437a268d7cf99756170f62",
  "meta": {
    "emotion": "neutral",
    "intensity": "normal",
    "statement": "Kids are talking by the door",
    "repetition": "1",
    "actor": "3",
    "gender": "male"
  }
}
```

Notes:
- `id` currently equals `relpath` and is a convenient UI row key.
- `h` is a fast, deterministic hash of `(relpath, size, mtime)` used to join with results cache. It is not content-invariant.
- `meta` contains per-file CSV fields, with redundant constants and duplicate `filename` removed (see below).
- `sample_rate` is provided when the WAV header contains it (top-level convenience; not part of `meta`).

## Summary Schema (dataset summary)

Returned by `GET /datasets/summary`:

```json
{
  "total": 144,
  "total_bytes": 12345678,
  "label_counts": {"happy": 30, "sad": 20, "neutral": 50},
  "meta_constants": {"modality": "audio-only", "vocal_channel": "speech"}
}
```

- `meta_constants` contains CSV columns whose values are identical for all rows. These are removed from each entry’s `meta` to reduce payload size.

## Results Cache

- GET: `/results/{model}/{h}`
- POST: `/results/{model}/{h}` with a JSON payload to cache
- Batch GET: `/results/{model}/batch` with body `{ "hashes": ["h1", "h2", ...] }`
- Batch response shape:

```json
{
  "ok": true,
  "payloads": {
    "h1": {"probabilities": {...}},
    "h2": null
  }
}
```

## Why filename, relpath, id, and h?

- `filename`: display-only basename
- `relpath`: used by `GET /datasets/file?relpath=...` to stream the file
- `id`: UI row key; currently equals `relpath` (can be dropped or remapped if desired)
- `h`: join key for results cache; efficient for `/results/...` endpoints

## Windows curl tips (quoting JSON)

- Command Prompt (cmd.exe):

```cmd
curl -s -X POST http://localhost:8000/datasets/reindex ^
  -H "Content-Type: application/json" ^
  -d "{\"id\":\"ravdess_subset\"}"
```

- PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/datasets/reindex -Method Post `
  -ContentType 'application/json' -Body '{"id":"ravdess_subset"}'
```

- File-based (works everywhere):

```cmd
echo {"id":"ravdess_subset"} > data.json
curl -s -X POST http://localhost:8000/datasets/reindex -H "Content-Type: application/json" --data-binary @data.json
```

## Frontend Notes

- The `/datasets/files` response now includes `meta` per entry (pruned) and the `/datasets/summary` includes `meta_constants`. Frontend can display constants once and show per-file fields under each row.

## Model In-Memory Cache (Per-Process)

- Models are cached in-process (module-level singletons) inside `Backend/app/services/model_loader_service.py`.
  - ASR (Whisper) pipelines: `_asr_pipelines` managed via `get_asr_pipeline(model_id)`.
  - Emotion (wav2vec2) components: `_emotion_feature_extractor`, `_emotion_model` via `get_emotion_components()`.
- Initialization is lazy and protected by per-model/thread locks to avoid duplicate loads under concurrency.
- Scope: per Python process. If you run multiple Uvicorn workers or containers, each holds its own model copy.
- Reset: restarting the process clears the in-memory cache (models will be recreated on first use).

### Startup Warmup (Optional)

- Implemented in `Backend/app/main.py` under `@app.on_event("startup")`.
- Control via `PRELOAD_MODELS` env var (comma-separated):
  - `whisper-base`, `whisper-large`, `wav2vec2`
- Example (PowerShell):

```powershell
$env:PRELOAD_MODELS = "whisper-base,whisper-large,wav2vec2"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Memory Considerations

- Whisper-large is heavy; expect high RAM/VRAM usage. Each additional worker multiplies memory usage.
- Prefer a single worker for development, or smaller models (e.g., `whisper-base`) if constrained.
- GPU is used automatically when available; fallback to CPU otherwise (see `_device` in `model_loader_service.py`).

### Redis Separation

- Redis stores only inference results (small JSON) keyed by model + file hash.
- Model objects are not serialized to Redis; they reside in process RAM for speed.

### Hugging Face Cache Persistence

- Model weights are downloaded and cached on disk by Hugging Face. Persist this cache across restarts:
  - Set `HF_HOME` to a stable path.
  - If using Docker, mount it as a volume.

Example (docker-compose service excerpt):

```yaml
environment:
  - HF_HOME=/cache/huggingface
volumes:
  - hf-cache:/cache/huggingface

volumes:
  hf-cache:
```

### Multi-Worker/Container Deployments

- Each worker/container will hold its own in-memory copy of models.
- This improves isolation but increases total memory use. Scale workers with care.

## Future Options

- Add a `content_hash` (SHA1 over file bytes) if you need prediction reuse across renames/moves. Keep the current `h` for speed and add `content_hash` only where needed.
