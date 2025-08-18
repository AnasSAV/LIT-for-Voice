import pytest
from httpx import AsyncClient

# Make app importable
from app.main import app
from app.core import redis as redis_module

@pytest.fixture(autouse=True, scope="function")
async def redis_cleanup():
    """Ensure a clean Redis database before and after each test.

    Requires a Redis server available at app.core.settings.REDIS_URL
    (default redis://localhost:6379/0). Start via docker-compose before tests.
    """
    # Flush before
    await redis_module.redis.flushall()
    try:
        yield
    finally:
        # Flush after
        await redis_module.redis.flushall()

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as ac:
        yield ac
