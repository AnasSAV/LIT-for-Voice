from fastapi import APIRouter, HTTPException, Query
import inspect
import asyncio
from pathlib import Path
from app.services.model_loader_service import *
router = APIRouter()



MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec
}

@router.get("/inferences/run")
async def run_inference(model: str, file_path: str = Query(None, description="Path to uploaded audio file")):
    print("calling infer api")
    func = MODEL_FUNCTIONS.get(model)

    if not func:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    # If file_path is provided, check if it exists
    if file_path:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {file_path}")

    # Detect if function is async or sync
    if inspect.iscoroutinefunction(func):
        if file_path:
            prediction = await func(file_path)  # Async function with file path
        else:
            prediction = await func()  # Async function with default file
    else:
        if file_path:
            prediction = await asyncio.to_thread(func, file_path)  # Sync function with file path in thread pool
        else:
            prediction = await asyncio.to_thread(func)  # Sync function with default file in thread pool

    return prediction
