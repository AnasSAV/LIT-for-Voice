import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.session import SessionMiddleware
from .services.model_loader_service import get_asr_pipeline, get_emotion_components
from .api.routes import (
    session as session_routes,
    results as results_routes,
    health as health_routes,
    datasets as datasets_routes,
    inferences as inferences_routes,
    upload as upload_routes
)

app = FastAPI(title="LIT for Voice – API")
origins = [
    "http://localhost:8080",  # your Vue/React dev server
    "http://localhost:5173",  # if using Vite
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # ["*"] for all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware)

# Include all routers
app.include_router(session_routes.router)
app.include_router(results_routes.router)
app.include_router(health_routes.router)
app.include_router(datasets_routes.router)
app.include_router(inferences_routes.router)
app.include_router(upload_routes.router)

@app.on_event("startup")
async def warmup_models():
    """Optionally warm up models on startup.
    Set PRELOAD_MODELS to a comma-separated list, e.g.:
    PRELOAD_MODELS=whisper-base,whisper-large,wav2vec2
    """
    raw = os.getenv("PRELOAD_MODELS", "").strip()
    if not raw:
        return
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    for t in tokens:
        try:
            if t in ("whisper-base", "openai/whisper-base"):
                get_asr_pipeline("openai/whisper-base")
            elif t in ("whisper-large", "openai/whisper-large-v3"):
                get_asr_pipeline("openai/whisper-large-v3")
            elif t in ("wav2vec2", "emotion", "wav2vec2-emotion"):
                get_emotion_components()
        except Exception:
            # Warmup is best-effort; avoid failing app startup
            pass

@app.get("/")
async def root():
    return {
        "message": "LIT for Voice API is running",
        "version": "1.0.0",
        "endpoints": ["/health", "/session", "/queue", "/results", "/inferences", "/upload"]
    }
