from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.session import SessionMiddleware
from .api.routes import session as session_routes, results as results_routes

app = FastAPI(title="LIT for Voice – API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # add prod FE
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(SessionMiddleware)

app.include_router(session_routes.router)
app.include_router(results_routes.router)
