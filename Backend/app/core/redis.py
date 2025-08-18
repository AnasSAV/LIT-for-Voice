import asyncio
import json
import uuid
from typing import Any, Optional
from .settings import settings

from redis.asyncio import Redis, from_url
from redis.exceptions import RedisError

# Initialize with None - will be created on first use
_redis: Optional[Redis] = None

def get_redis() -> Redis:
    """Get the Redis client instance, creating it if necessary."""
    global _redis
    if _redis is None:
        _redis = from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True
        )
    return _redis

def reset_client() -> None:
    """Reset the Redis client, closing any existing connection.
    
    This is particularly important in tests where the event loop may be recreated.
    """
    global _redis
    if _redis is not None:
        try:
            # Schedule the close operation in the background
            asyncio.create_task(_redis.aclose())
        except (RuntimeError, RedisError, AttributeError):
            # Ignore errors during cleanup or if aclose() is not available
            pass
    _redis = None

# For backward compatibility
redis = get_redis()

def k_sess(sid: str) -> str:  return f"sess:{sid}"
def k_queue(sid: str) -> str: return f"{k_sess(sid)}:queue"
def k_meta(sid: str) -> str:  return f"{k_sess(sid)}:meta"
def k_result(model: str, h: str) -> str: return f"result:{model}:{h}"
def k_ds_manifest(ds_id: str) -> str: return f"dataset:{ds_id}:manifest"
def k_ds_summary(ds_id: str) -> str:  return f"dataset:{ds_id}:summary"
def k_ds_version(ds_id: str) -> str:  return f"dataset:{ds_id}:version"

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
