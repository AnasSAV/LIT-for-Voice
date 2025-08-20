from fastapi import APIRouter, HTTPException, Query, Body
import inspect
import asyncio
from pathlib import Path
from app.services.model_loader_service import *
from app.services.dataset_manifest import get_or_build_manifest
import hashlib
import json
from app.core.redis import get_result, cache_result
router = APIRouter()



MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec
}

def _params_hash(params: dict | None) -> str:
    try:
        return hashlib.sha1(json.dumps(params or {}, sort_keys=True).encode()).hexdigest()
    except Exception:
        return "default"


def _file_bytes_hash(p: Path) -> str | None:
    try:
        data = p.read_bytes()
        return hashlib.sha1(data).hexdigest()
    except Exception:
        return None


@router.get("/inferences/run")
async def run_inference(
    model: str,
    file_path: str = Query(None, description="Path to uploaded audio file"),
    ds_id: str | None = Query(None, description="Dataset id if file is from a dataset"),
    h: str | None = Query(None, description="Dataset manifest file hash for caching"),
):
    print("calling infer api")
    func = MODEL_FUNCTIONS.get(model)

    if not func:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    # If dataset id and hash provided but no file_path, resolve absolute file path via manifest
    if (not file_path) and ds_id and h:
        # Resolve dataset base directory (restricted to subset datasets)
        backend_root = Path(__file__).resolve().parents[3]
        data_root = backend_root / "data"
        if ds_id == "ravdess_subset":
            base = data_root / "dev" / "ravdess_subset"
        elif ds_id == "common_voice_en_dev":
            base = data_root / "dev" / "common_voice_en_dev"
        else:
            base = Path("")
        if not base.exists():
            raise HTTPException(status_code=404, detail=f"Dataset path not found for id: {ds_id}")
        entries, _ = await get_or_build_manifest(ds_id, base)
        match = next((e for e in entries if e.get("h") == h), None)
        if not match:
            raise HTTPException(status_code=404, detail="File hash not found in dataset manifest")
        abs_path = (base / match.get("relpath", "")).resolve()
        if not abs_path.exists():
            raise HTTPException(status_code=404, detail="Resolved audio file not found on disk")
        file_path = str(abs_path)

    # If file_path is provided (either directly or resolved), check if it exists
    if file_path:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {file_path}")

    # Build cache key (model + input hash + params hash). No extra params yet, so default.
    params_h = _params_hash(None)
    cache_h: str | None = None
    if ds_id and h:
        cache_h = f"{ds_id}:{h}:{params_h}"
    elif file_path:
        fb = _file_bytes_hash(Path(file_path))
        if fb:
            cache_h = f"bytes:{fb}:{params_h}"

    # Try cache if we have a key
    if cache_h:
        cached = await get_result(model, cache_h)
        if cached is not None:
            return cached

    # Detect if function is async or sync
    if inspect.iscoroutinefunction(func):
        if file_path:
            prediction = await func(file_path)  # Async function with file path
        else:
            prediction = await func()  # Async function with default file
    else:
        if file_path:
            prediction = await asyncio.to_thread(func, file_path)  # Sync function with file path in thread pool
        else:
            prediction = await asyncio.to_thread(func)  # Sync function with default file in thread pool

    # Cache the result for future reuse
    if cache_h:
        try:
            await cache_result(model, cache_h, prediction)  # prediction can be dict or str
        except Exception:
            pass

    return prediction


@router.post("/inferences/flush")
async def flush_models(payload: dict = Body(None)):
    """
    Flush cached models to free RAM/VRAM.
    Payload (optional): {"model": "whisper-base" | "whisper-large" | "wav2vec2" | "all"}
    If omitted or "all", unloads all cached models.
    """
    try:
        target = None
        if isinstance(payload, dict):
            target = payload.get("model")

        # Map UI/route model names to HF ids where needed
        if target in (None, "all"):
            summary = unload_all_models()
            return {"ok": True, "scope": "all", "summary": summary}

        if target == "whisper-base":
            cnt = unload_asr_model("openai/whisper-base")
            return {"ok": True, "scope": target, "asr_removed": cnt}
        if target == "whisper-large" or target == "whisper-large-v3":
            cnt = unload_asr_model("openai/whisper-large-v3")
            return {"ok": True, "scope": target, "asr_removed": cnt}
        if target == "wav2vec2":
            removed = unload_emotion_model()
            return {"ok": True, "scope": target, "emotion_removed": removed}

        raise HTTPException(status_code=400, detail=f"Unknown model scope: {target}")
    except HTTPException:
        raise
    except Exception as e:
        # Avoid leaking internal errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inferences/cache_status")
async def cache_status():
    try:
        status = get_cache_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
