from fastapi import APIRouter, Body
from ...core.redis import get_result, cache_result, redis, k_result
import json

router = APIRouter()

@router.get("/results/{model}/{h}")
async def results_get(model: str, h: str):
    payload = await get_result(model, h)
    return {"cached": payload is not None, "payload": payload}

@router.post("/results/{model}/{h}")
async def results_put(model: str, h: str, payload: dict = Body(...)):
    await cache_result(model, h, payload)
    return {"ok": True}


@router.post("/results/{model}/batch")
async def results_batch(model: str, payload: dict = Body(...)):
    """
    Fetch cached results for multiple hashes in one call.
    Body: {"hashes": ["h1", "h2", ...]}
    Returns a mapping from hash to payload (or null if missing).
    """
    hashes = payload.get("hashes") if isinstance(payload, dict) else None
    if not isinstance(hashes, list):
        return {"ok": False, "error": "hashes must be a list"}

    pipe = redis.pipeline()
    for h in hashes:
        pipe.get(k_result(model, str(h)))
    raws = await pipe.execute()

    out = {}
    for h, raw in zip(hashes, raws):
        try:
            out[str(h)] = json.loads(raw) if raw else None
        except Exception:
            out[str(h)] = None

    return {"ok": True, "payloads": out}
