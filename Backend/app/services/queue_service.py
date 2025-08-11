from ..core import redis as r

async def add_item(sid: str, item: dict):
    s = await r.get_queue(sid)
    s.setdefault("items", []).append(item)
    await r.put_queue(sid, s)
    return s

async def set_progress(sid: str, update: dict):
    s = await r.get_queue(sid)
    s.update(update)
    await r.put_queue(sid, s)
    return s
