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

## Future Options

- Add a `content_hash` (SHA1 over file bytes) if you need prediction reuse across renames/moves. Keep the current `h` for speed and add `content_hash` only where needed.
