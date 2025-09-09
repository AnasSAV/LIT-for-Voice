from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging

router = APIRouter(prefix="/dataset-files", tags=["dataset-files"])
logger = logging.getLogger(__name__)

# Resolve paths relative to repo structure: Backend/data/
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAVDESS_DIR = DATA_DIR / "ravdess_subset"
COMMON_VOICE_DIR = DATA_DIR / "common_voice_valid_dev"

@router.get("/{dataset}/{filename}")
async def serve_dataset_audio_file(dataset: str, filename: str):
    """
    Serve audio files from datasets for playback
    """
    if dataset not in ["ravdess", "common-voice"]:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
    
    if dataset == "ravdess":
        audio_dir = RAVDESS_DIR
    else:
        audio_dir = COMMON_VOICE_DIR
    
    file_path = audio_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    # Determine the correct media type based on file extension
    file_extension = file_path.suffix.lower()
    media_type_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac'
    }
    media_type = media_type_map.get(file_extension, 'audio/*')
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'public, max-age=3600',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range, Accept-Encoding',
            'Content-Disposition': f'inline; filename="{filename}"'
        }
    )
