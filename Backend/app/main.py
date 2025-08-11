from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.session import SessionMiddleware
from .api.routes import session as session_routes, results as results_routes
# ➕ imports for health
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from .core.redis import redis

app = FastAPI(title="LIT for Voice – API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # add your FE origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware)

app.include_router(session_routes.router)
app.include_router(results_routes.router)

# ➕ health check (simple and fast)
@app.get("/health")
async def health():
    try:
        pong = await redis.ping()
        return {"status": "ok", "redis": bool(pong)}
    except RedisError as e:
        # Return 503 if Redis isn’t reachable
        return JSONResponse({"status": "degraded", "redis": False, "detail": str(e)}, status_code=503)
