from fastapi import APIRouter, HTTPException, Query
import inspect
import asyncio
import logging
from pathlib import Path
from typing import Optional
from app.services.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    wave2vec,
)
from app.services.dataset_service import resolve_file

router = APIRouter()
logger = logging.getLogger(__name__)


MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec,
}


@router.get("/inferences/run")
async def run_inference(
    model: str,
    file_path: Optional[str] = Query(None, description="Path to uploaded audio file (server-visible)"),
    dataset: Optional[str] = Query(None, description="Dataset key for server-side files (e.g., 'common-voice', 'ravdess')"),
    dataset_file: Optional[str] = Query(None, description="Filename within dataset (basename only)"),
):
    logger.info(
        "inferences.run model=%s file_path=%s dataset=%s dataset_file=%s",
        model,
        file_path,
        dataset,
        dataset_file,
    )

    func = MODEL_FUNCTIONS.get(model)
    if not func:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    resolved_path: Optional[Path] = None

    if file_path:
        resolved_path = Path(file_path)
    elif dataset and dataset_file:
        try:
            # Resolve using service (enforces allowed datasets and basename-only)
            resolved_path = resolve_file(dataset, dataset_file)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            # Unknown dataset or other
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'.",
        )

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")

    # Detect if function is async or sync and call appropriately
    if inspect.iscoroutinefunction(func):
        prediction = await func(str(resolved_path))
    else:
        prediction = await asyncio.to_thread(func, str(resolved_path))

    return prediction
