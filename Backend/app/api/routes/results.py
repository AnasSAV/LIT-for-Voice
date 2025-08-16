from fastapi import APIRouter, Body
from ...core.redis import get_result, cache_result

router = APIRouter()

@router.get("/results/{model}/{h}")
async def results_get(model: str, h: str):
    payload = await get_result(model, h)
    return {"cached": payload is not None, "payload": payload}

@router.post("/results/{model}/{h}")
async def results_put(model: str, h: str, payload: dict = Body(...)):
    await cache_result(model, h, payload)
    return {"ok": True}
