from fastapi import APIRouter, Request, Body
from ...core import redis as r
from ...services.queue_service import add_item, set_progress

router = APIRouter()

@router.get("/session")
async def session_info(req: Request): return {"sid": req.state.sid}

@router.get("/queue")
async def queue_get(req: Request): return await r.get_queue(req.state.sid)

@router.post("/queue/add")
async def queue_add(req: Request, item: dict = Body(...)): return {"state": await add_item(req.state.sid, item)}

@router.patch("/queue/progress")
async def queue_progress(req: Request, update: dict = Body(...)): return {"state": await set_progress(req.state.sid, update)}

@router.delete("/queue")
async def queue_clear(req: Request):
    await r.put_queue(req.state.sid, {"items": [], "processing": None, "completed": []})
    return {"ok": True}
