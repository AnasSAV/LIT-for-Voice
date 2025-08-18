import pytest
import httpx

from app.main import app
from app.core import redis as redis_module

@pytest.fixture(autouse=True, scope="function")
async def redis_cleanup():
    if redis_module.redis:
        try:
            await redis_module.redis.aclose()
        except (RuntimeError, AttributeError):
            pass
    
    redis_module.reset_client()
    try:
        await redis_module.redis.ping()
        await redis_module.redis.flushall()
        yield
    finally:
        try:
            if redis_module.redis:
                await redis_module.redis.flushall()
                await redis_module.redis.aclose()
        except RuntimeError:
            pass
        finally:
            redis_module.reset_client()

@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as ac:
        yield ac
