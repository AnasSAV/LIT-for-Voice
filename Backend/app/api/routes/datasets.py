from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from app.services.dataset_service import (
    load_metadata,
    resolve_file,
    media_type_for,
)
router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{dataset}/metadata")
async def get_dataset_metadata(dataset: str) -> JSONResponse:
    try:
        rows: List[dict] = load_metadata(dataset)
        return JSONResponse(content=rows)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")


@router.get("/{dataset}/file/{file_path:path}")
@router.head("/{dataset}/file/{file_path:path}")
@router.options("/{dataset}/file/{file_path:path}")
async def serve_dataset_file(dataset: str, file_path: str):
    try:
        audio_path = resolve_file(dataset, file_path)
    except ValueError as e:
        # Unknown dataset
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        media_type = media_type_for(audio_path)
    except ValueError as e:
        # Unsupported media type
        raise HTTPException(status_code=415, detail=str(e))

    safe_name = audio_path.name
    return FileResponse(
        path=audio_path,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Accept-Encoding",
            "Content-Disposition": f"inline; filename=\"{safe_name}\"",
        },
    )
