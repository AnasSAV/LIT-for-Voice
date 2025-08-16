from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import inspect
import asyncio
import tempfile
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

from app.services.model_loader_service import *
from app.services.model_loader_service import _model_cache
from app.core.redis import cache_result, get_result, clear_model_cache

router = APIRouter()
logger = logging.getLogger(__name__)

# Sample file functions for backward compatibility
MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec
}

# File upload functions
FILE_MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base_file,
    "whisper-large": transcribe_whisper_large_file,
    "wav2vec2": wave2vec_file
}

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for uploaded file content"""
    return hashlib.md5(file_content).hexdigest()

async def get_cached_or_compute(cache_key: str, model: str, compute_func, *args, **kwargs):
    """Get result from cache or compute and cache it"""
    
    # Try to get from cache first
    cached_result = await get_result(model, cache_key)
    if cached_result:
        logger.info(f"Cache hit for {model}:{cache_key}")
        return cached_result
    
    # Compute result
    logger.info(f"Cache miss for {model}:{cache_key}, computing...")
    if inspect.iscoroutinefunction(compute_func):
        result = await compute_func(*args, **kwargs)
    else:
        result = await asyncio.to_thread(compute_func, *args, **kwargs)
    
    # Cache the result
    cache_payload = {"prediction": result}
    await cache_result(model, cache_key, cache_payload, ttl=24*60*60)  # Cache for 24 hours
    
    return cache_payload

@router.get("/inferences/run")
async def run_inference(model: str):
    """Run inference on sample files with caching"""
    func = MODEL_FUNCTIONS.get(model)
    
    if not func:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    
    # Use model name as cache key for sample files since they don't change
    cache_key = f"sample_{model}"
    
    try:
        result = await get_cached_or_compute(cache_key, model, func)
        logger.info(f"Inference completed for model: {model}")
        return result
    except Exception as e:
        logger.error(f"Error running inference for {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")

@router.post("/inferences/upload")
async def run_inference_upload(
    model: str = Form(...),
    file: UploadFile = File(...)
):
    """Run inference on uploaded audio file with caching"""
    
    # Validate model
    func = FILE_MODEL_FUNCTIONS.get(model)
    if not func:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Generate cache key based on file content and model
        file_hash = get_file_hash(file_content)
        cache_key = f"upload_{file_hash}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Get cached result or compute
            result = await get_cached_or_compute(cache_key, model, func, temp_file_path)
            logger.info(f"Inference completed for uploaded file with model: {model}")
            return result
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                logger.warning(f"Failed to delete temporary file: {temp_file_path}")
                
    except Exception as e:
        logger.error(f"Error processing uploaded file with {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/inferences/models")
async def list_available_models():
    """List all available models"""
    return {
        "models": list(MODEL_FUNCTIONS.keys()),
        "description": {
            "whisper-base": "OpenAI Whisper Base - Fast speech transcription",
            "whisper-large": "OpenAI Whisper Large - High-quality speech transcription", 
            "wav2vec2": "Wav2Vec2 - Emotion recognition from speech"
        }
    }

@router.delete("/inferences/cache/{model}")
async def clear_model_cache_endpoint(model: str):
    """Clear cache for a specific model (admin endpoint)"""
    if model not in MODEL_FUNCTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    
    try:
        cleared_count = await clear_model_cache(model)
        return {
            "message": f"Cache cleared for model: {model}",
            "cleared_entries": cleared_count
        }
    except Exception as e:
        logger.error(f"Error clearing cache for {model}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@router.get("/inferences/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(_model_cache),
        "available_models": list(MODEL_FUNCTIONS.keys())
    }
