from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List
import os
from app.services.dataset_service import (
    load_metadata,
    resolve_file,
    media_type_for,
)
router = APIRouter()
logger = logging.getLogger(__name__)


def get_session_id(request: Request) -> str:
    """Extract session ID from request (optional for backwards compatibility)"""
    return getattr(request.state, 'sid', None)


@router.get("/{dataset}/metadata")
async def get_dataset_metadata(dataset: str, request: Request) -> JSONResponse:
    try:
        session_id = get_session_id(request)
        rows: List[dict] = load_metadata(dataset, session_id)
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
async def serve_dataset_file(dataset: str, file_path: str, request: Request):
    try:
        session_id = get_session_id(request)
        audio_path = resolve_file(dataset, file_path, session_id)
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
    file_size = audio_path.stat().st_size
    
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        return JSONResponse(
            content="",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Range, Accept-Encoding, Origin, X-Requested-With, Content-Type, Accept, Authorization",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    
    # Handle Range requests for better streaming support
    range_header = request.headers.get('range')
    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        try:
            ranges = range_header.replace('bytes=', '').split('-')
            start = int(ranges[0]) if ranges[0] else 0
            end = int(ranges[1]) if ranges[1] else file_size - 1
            
            # Ensure valid range
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            content_length = end - start + 1
            
            def generate_chunks():
                with open(audio_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            headers = {
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Range, Accept-Encoding, Origin, X-Requested-With, Content-Type, Accept, Authorization",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
                "Content-Disposition": f"inline; filename=\"{safe_name}\"",
                "X-Content-Type-Options": "nosniff",
            }
            
            return StreamingResponse(
                generate_chunks(),
                status_code=206,  # Partial Content
                media_type=media_type,
                headers=headers
            )
        except (ValueError, IndexError):
            # Invalid range header, fall back to full file
            pass
    
    # Return full file for non-range requests
    return FileResponse(
        path=audio_path,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Accept-Encoding, Origin, X-Requested-With, Content-Type, Accept, Authorization",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
            "Content-Disposition": f"inline; filename=\"{safe_name}\"",
            "X-Content-Type-Options": "nosniff",
        },
    )
