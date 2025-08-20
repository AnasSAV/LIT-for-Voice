from fastapi import APIRouter, Request, Body
from pathlib import Path
from ...core.redis import redis, k_meta
from typing import List, Optional
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse
from ...services.dataset_manifest import get_or_build_manifest

router = APIRouter()

# Resolve Backend root and data paths
BACKEND_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = BACKEND_ROOT / "data"
DEV_RAVDESS = DATA_ROOT / "dev" / "ravdess_subset"
DEV_COMMON_VOICE_EN = DATA_ROOT / "dev" / "common_voice_en_dev"

def _available_datasets():
    return [
        {
            "id": "ravdess_subset",
            "name": "RAVDESS (Subset)",
            "available": DEV_RAVDESS.exists(),
            "path": str(DEV_RAVDESS),
        },
        {
            "id": "common_voice_en_dev",
            "name": "Common Voice (en) Dev Subset",
            "available": DEV_COMMON_VOICE_EN.exists(),
            "path": str(DEV_COMMON_VOICE_EN),
        },
    ]


def _default_dataset_id() -> Optional[str]:
    """Prefer a subset dataset by default if available."""
    if DEV_RAVDESS.exists():
        return "ravdess_subset"
    if DEV_COMMON_VOICE_EN.exists():
        return "common_voice_en_dev"
    return None


@router.get("/datasets")
async def list_datasets():
    return {"datasets": _available_datasets()}


@router.get("/datasets/active")
async def get_active_dataset(req: Request):
    sid = req.state.sid
    # Read from session meta
    active = await redis.hget(k_meta(sid), "active_dataset")
    if not active:
        # Set a default (RAVDESS) if available
        default_id = _default_dataset_id()
        if default_id:
            active = default_id
            await redis.hset(k_meta(sid), mapping={"active_dataset": active})
    return {"active": active}


@router.post("/datasets/select")
async def select_dataset(req: Request, payload: dict = Body(...)):
    sid = req.state.sid
    ds_id = payload.get("id")

    # Validate requested dataset
    valid_ids = {d["id"] for d in _available_datasets()}
    if ds_id not in valid_ids:
        return {"ok": False, "error": "Unknown dataset", "valid": sorted(list(valid_ids))}

    # Persist selection in session meta
    await redis.hset(k_meta(sid), mapping={"active_dataset": ds_id})
    return {"ok": True, "active": ds_id}


def _dataset_path_for_id(ds_id: str) -> Path:
    if ds_id == "ravdess_subset":
        return DEV_RAVDESS
    if ds_id == "common_voice_en_dev":
        return DEV_COMMON_VOICE_EN
    return Path("")


@router.get("/datasets/files")
async def list_dataset_files(req: Request, limit: int = 100, offset: int = 0):
    """
    List audio files for the active dataset. Returns filenames and relative paths.
    """
    sid = req.state.sid
    active = await redis.hget(k_meta(sid), "active_dataset")
    if not active:
        # choose and set default if available
        default_id = _default_dataset_id()
        if default_id:
            active = default_id
            await redis.hset(k_meta(sid), mapping={"active_dataset": active})
        else:
            return {"total": 0, "files": [], "active": None}

    base = _dataset_path_for_id(active)
    if not base.exists():
        return {"total": 0, "files": [], "active": active}

    # Use cached manifest
    entries, summary = await get_or_build_manifest(active, base)
    total = summary.get("total", len(entries))

    # window
    items = []
    for e in entries[offset: offset + limit]:
        items.append({
            "id": e.get("id"),
            "filename": e.get("filename"),
            "relpath": e.get("relpath"),
            "size": e.get("size"),
            "duration": e.get("duration"),
            "label": e.get("label"),
            "h": e.get("h"),
            "meta": e.get("meta"),
        })

    return {"total": total, "files": items, "active": active}


@router.get("/datasets/file")
async def get_dataset_file(
    req: Request,
    relpath: str = Query(..., description="Relative path within the active dataset"),
    id: Optional[str] = Query(None, description="Dataset id override: ravdess_subset or common_voice_en_dev"),
):
    # Determine dataset base dir
    ds_id = id
    if not ds_id:
        sid = req.state.sid
        ds_id = await redis.hget(k_meta(sid), "active_dataset")
        if not ds_id:
            default_id = _default_dataset_id()
            if default_id:
                ds_id = default_id
                await redis.hset(k_meta(sid), mapping={"active_dataset": ds_id})
            else:
                raise HTTPException(status_code=400, detail="No active dataset selected")

    base = _dataset_path_for_id(ds_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="Dataset path not found")

    # Resolve and ensure the requested file stays within base directory
    abs_path = (base / relpath).resolve()
    try:
        abs_path.relative_to(base.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Serve a correct content type based on extension
    ext = abs_path.suffix.lower()
    if ext == ".mp3":
        media_type = "audio/mpeg"
    elif ext == ".wav":
        media_type = "audio/wav"
    else:
        media_type = "application/octet-stream"

    return FileResponse(path=str(abs_path), media_type=media_type, filename=abs_path.name)


@router.get("/datasets/summary")
async def dataset_summary(req: Request, id: Optional[str] = Query(None)):
    """
    Return cached summary for the active dataset, or a specified dataset id.
    Summary includes at least: total, total_bytes (and future label_counts).
    """
    ds_id = id
    if not ds_id:
        sid = req.state.sid
        ds_id = await redis.hget(k_meta(sid), "active_dataset")
        if not ds_id:
            default_id = _default_dataset_id()
            if default_id:
                ds_id = default_id
                await redis.hset(k_meta(sid), mapping={"active_dataset": ds_id})
            else:
                return {"active": None, "summary": {"total": 0, "total_bytes": 0}}

    base = _dataset_path_for_id(ds_id)
    if not base.exists():
        return {"active": ds_id, "summary": {"total": 0, "total_bytes": 0}}

    _, summary = await get_or_build_manifest(ds_id, base)
    return {"active": ds_id, "summary": summary}


@router.post("/datasets/reindex")
async def reindex_dataset(payload: dict = Body(...)):
    """
    Force a manifest rebuild for a given dataset id. Dev/admin utility.
    Payload: {"id": "ravdess_subset" | "common_voice_en_dev"}
    """
    ds_id = payload.get("id")
    valid_ids = {d["id"] for d in _available_datasets()}
    if ds_id not in valid_ids:
        return {"ok": False, "error": "Unknown dataset id", "valid": sorted(list(valid_ids))}

    base = _dataset_path_for_id(ds_id)
    if not base.exists():
        return {"ok": False, "error": "Dataset path not found", "id": ds_id}

    entries, summary = await get_or_build_manifest(ds_id, base, force=True)
    return {"ok": True, "id": ds_id, "total": summary.get("total", len(entries))}
