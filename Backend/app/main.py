from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.session import SessionMiddleware
from .api.routes import session as session_routes, results as results_routes
from .api.routes import health as health_routes
from .api.routes import datasets as datasets_routes

app = FastAPI(title="LIT for Voice – API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],  # add your FE origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware)
app.include_router(session_routes.router)
app.include_router(results_routes.router)
app.include_router(health_routes.router)
app.include_router(datasets_routes.router)

@app.get("/")
async def root():
    return {"message": "LIT for Voice API is running", "version": "1.0.0", "endpoints": ["/health", "/session", "/queue", "/results"]}

