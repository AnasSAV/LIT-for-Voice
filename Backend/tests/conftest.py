import asyncio
import os
import pytest
from httpx import AsyncClient
from fakeredis.aioredis import FakeRedis

# Make app importable
from app.main import app
from app.core import redis as redis_module

@pytest.fixture(autouse=True, scope="function")
async def fake_redis(monkeypatch):
    """
    Replace the global redis client with fakeredis for each test.
    """
    client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_module, "redis", client)
    yield
    await client.flushall()
    await client.aclose()

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as ac:
        yield ac
