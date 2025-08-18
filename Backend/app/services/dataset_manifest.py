import asyncio
import json
import hashlib
import contextlib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import wave
import csv

from ..core.redis import (
    redis,
    k_ds_manifest,
    k_ds_summary,
    k_ds_version,
)
from ..core.settings import settings

# Public API
# - compute_dir_version(base)
# - get_or_build_manifest(ds_id, base, ttl=86400, force=False)


def _posix_rel(base: Path, p: Path) -> str:
    try:
        rel = p.relative_to(base)
    except Exception:
        rel = p.name
    return str(rel).replace("\\", "/")


def _compute_file_hash(relpath: str, size: int, mtime: int) -> str:
    h = hashlib.sha1()
    h.update(relpath.encode("utf-8"))
    h.update(str(size).encode("utf-8"))
    h.update(str(mtime).encode("utf-8"))
    return h.hexdigest()


def _safe_duration_wav(p: Path) -> Optional[float]:
    # Compute duration for standard PCM WAV files using stdlib
    with contextlib.suppress(Exception):
        with wave.open(str(p), "rb") as wf:
            framerate = wf.getframerate() or 0
            nframes = wf.getnframes() or 0
            if framerate > 0:
                return float(nframes) / float(framerate)
    return None


def _parse_label_from_filename(filename: str) -> Optional[str]:
    """
    Parse ground-truth label from filename when possible.
    - Supports RAVDESS pattern: MM-VC-EM-IT-ST-RP-AC.wav
      EM (emotion) codes:
        01 neutral, 02 calm, 03 happy, 04 sad, 05 angry,
        06 fearful, 07 disgust, 08 surprised
    Returns the emotion string when recognized; else None.
    """
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = name
    if "." in name:
        stem = name[: name.rfind(".")]
    parts = stem.split("-")
    if len(parts) == 7 and all(p.isdigit() for p in parts):
        emotion_code = parts[2]
        em_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised",
        }
        return em_map.get(emotion_code)
    return None


def _load_metadata_csv(base: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load optional dataset-level metadata from a CSV residing in `base`.
    We look for the first file matching '*_metadata.csv'. Expected to contain
    at least a 'filename' column. Returns mapping: filename -> row dict.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    try:
        for p in sorted(base.glob("*_metadata.csv")):
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Common column names: 'filename' | 'file' | 'name'
                    key = (row.get("filename") or row.get("file") or row.get("name") or "").strip()
                    if key:
                        mapping[key] = row
            break  # only the first match is used
    except Exception:
        # Metadata is optional; silently ignore issues
        pass
    return mapping


def _metadata_constants(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return columns whose values are identical (and non-empty) for all rows."""
    if not rows:
        return {}
    keys: set = set()
    for r in rows.values():
        keys.update(r.keys())
    constants: Dict[str, Any] = {}
    for k in keys:
        # collect values; if any missing/empty, not a constant
        vals = []
        complete = True
        for r in rows.values():
            v = r.get(k)
            if v in (None, ""):
                complete = False
                break
            vals.append(v)
        if complete and len(set(vals)) == 1:
            constants[k] = vals[0]
    return constants


def compute_dir_version(base: Path) -> str:
    """
    Build a content snapshot hash based on relpath, size, mtime for all wav files.
    """
    if not base.exists():
        return "empty"
    records: List[Tuple[str, int, int]] = []
    for p in sorted(base.rglob("*.wav")):
        try:
            st = p.stat()
            rel = _posix_rel(base, p)
            records.append((rel, int(st.st_size), int(st.st_mtime)))
        except Exception:
            continue
    # Also include metadata CSV files so cache invalidates when they change
    for p in sorted(base.glob("*_metadata.csv")):
        try:
            st = p.stat()
            rel = _posix_rel(base, p)
            records.append((rel, int(st.st_size), int(st.st_mtime)))
        except Exception:
            continue
    snap = hashlib.sha1()
    for rel, size, mtime in records:
        snap.update(rel.encode("utf-8"))
        snap.update(str(size).encode("utf-8"))
        snap.update(str(mtime).encode("utf-8"))
    return snap.hexdigest()


async def _scan_dataset(base: Path, ds_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Scan filesystem and compute attributes for all wav files.
    Returns (entries, summary, version).
    """
    if not base.exists():
        return [], {"total": 0, "total_bytes": 0}, "empty"

    entries: List[Dict[str, Any]] = []
    total_bytes = 0
    label_counts: Dict[str, int] = {}

    # Load optional dataset metadata to enrich entries
    metadata_by_filename = _load_metadata_csv(base)
    meta_consts = _metadata_constants(metadata_by_filename)
    # Prune duplicates from per-entry meta; also drop duplicate 'filename'
    drop_meta_keys = set(meta_consts.keys()) | {"filename"}

    for p in sorted(base.rglob("*.wav")):
        try:
            st = p.stat()
        except Exception:
            continue
        rel = _posix_rel(base, p)
        size = int(st.st_size)
        mtime = int(st.st_mtime)
        duration = await asyncio.to_thread(_safe_duration_wav, p)
        label = _parse_label_from_filename(p.name)
        h = _compute_file_hash(rel, size, mtime)

        total_bytes += size
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
        entry: Dict[str, Any] = {
            "id": rel,  # stable row id
            "relpath": rel,
            "filename": p.name,
            "size": size,
            "duration": duration,
            "label": label,
            "h": h,
        }
        # Attach CSV metadata if available for this filename
        meta_row = metadata_by_filename.get(p.name)
        if meta_row:
            filtered = {k: v for k, v in meta_row.items() if k not in drop_meta_keys}
            if filtered:
                entry["meta"] = filtered
        entries.append(entry)

    summary = {
        "total": len(entries),
        "total_bytes": total_bytes,
        "label_counts": label_counts,
    }
    if meta_consts:
        summary["meta_constants"] = meta_consts
    version = compute_dir_version(base)
    return entries, summary, version


async def get_or_build_manifest(
    ds_id: str,
    base: Path,
    ttl: Optional[int] = None,
    force: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Return (entries, summary) from Redis if valid, else build, cache, and return.
    """
    # Compute current directory version
    current_version = compute_dir_version(base)

    # Try to read existing cached data
    pipe = redis.pipeline()
    pipe.get(k_ds_manifest(ds_id))
    pipe.get(k_ds_summary(ds_id))
    pipe.get(k_ds_version(ds_id))
    cached_manifest_json, cached_summary_json, cached_version = await pipe.execute()

    if (
        not force
        and cached_manifest_json
        and cached_summary_json
        and cached_version == current_version
    ):
        try:
            return json.loads(cached_manifest_json), json.loads(cached_summary_json)
        except Exception:
            pass  # fall through to rebuild

    # Build fresh
    entries, summary, version = await _scan_dataset(base, ds_id)

    # Store in Redis with TTL
    effective_ttl = ttl if ttl is not None else settings.DATASET_CACHE_TTL_SECONDS
    pipe = redis.pipeline()
    pipe.set(k_ds_manifest(ds_id), json.dumps(entries), ex=effective_ttl)
    pipe.set(k_ds_summary(ds_id), json.dumps(summary), ex=effective_ttl)
    pipe.set(k_ds_version(ds_id), version, ex=effective_ttl)
    await pipe.execute()

    return entries, summary
