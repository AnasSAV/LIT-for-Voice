from fastapi import APIRouter, HTTPException, Query
import inspect
import asyncio
from pathlib import Path
from app.services.model_loader_service import *
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

    # If file_path is provided, check if it exists
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
