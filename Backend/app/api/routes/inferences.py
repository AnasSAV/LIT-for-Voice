from fastapi import APIRouter, HTTPException
import inspect
import asyncio
from app.services.model_loader_service import *
router = APIRouter()



MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec
}

@router.get("/inferences/run")
async def run_inference(model: str):
    func = MODEL_FUNCTIONS.get(model)

    if not func:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    # Detect if function is async or sync
    if inspect.iscoroutinefunction(func):
        prediction = await func()  # Async function
    else:
        prediction = await asyncio.to_thread(func)  # Sync function in thread pool

    print(prediction)
    return {"prediction": prediction}
