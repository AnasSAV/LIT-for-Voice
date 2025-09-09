import json, uuid
from typing import Any
from redis.asyncio import from_url
from .settings import settings

redis = from_url(settings.REDIS_URL, decode_responses=True)

def get_redis():
    """Get the Redis client instance"""
    return redis

def k_sess(sid: str) -> str:  return f"sess:{sid}"
def k_queue(sid: str) -> str: return f"{k_sess(sid)}:queue"
def k_meta(sid: str) -> str:  return f"{k_sess(sid)}:meta"
def k_result(model: str, h: str) -> str: return f"result:{model}:{h}"

async def ensure_session(sid: str | None) -> str:
    if not sid: sid = uuid.uuid4().hex
    p = redis.pipeline()
    p.hsetnx(k_meta(sid), "created", "1")
    p.expire(k_queue(sid), settings.SESSION_TTL_SECONDS)
    p.expire(k_meta(sid), settings.SESSION_TTL_SECONDS)
    await p.execute()
    return sid

async def get_queue(sid: str) -> dict[str, Any]:
    raw = await redis.get(k_queue(sid))
    return json.loads(raw) if raw else {"items": [], "processing": None, "completed": []}

async def put_queue(sid: str, state: dict[str, Any]) -> None:
    await redis.set(k_queue(sid), json.dumps(state), ex=settings.SESSION_TTL_SECONDS)

async def cache_result(model: str, h: str, payload: dict, ttl: int = 6*60*60) -> None:
    await redis.set(k_result(model, h), json.dumps(payload), ex=ttl)

async def get_result(model: str, h: str) -> dict | None:
    raw = await redis.get(k_result(model, h))
    return json.loads(raw) if raw else None
