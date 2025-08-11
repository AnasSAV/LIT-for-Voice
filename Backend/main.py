# main.py
from fastapi import FastAPI
from api.inferenceapi import router as inference_router

app = FastAPI()

app.include_router(inference_router, prefix="/api/inference")

# Optionally, a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the API"}
