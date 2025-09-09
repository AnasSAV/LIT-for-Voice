from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.session import SessionMiddleware

from .api.routes import session as session_routes, results as results_routes, inferences as inferences_routes, upload as upload_routes
from .api.routes import datasets as datasets_routes, predictions as predictions_routes, dataset_files as dataset_files_routes

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
# app.add_middleware(SessionMiddleware)

app.include_router(session_routes.router)
app.include_router(results_routes.router)
app.include_router(inferences_routes.router)
app.include_router(upload_routes.router)
app.include_router(datasets_routes.router)
app.include_router(predictions_routes.router)
app.include_router(dataset_files_routes.router)
