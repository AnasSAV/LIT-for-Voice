from fastapi import APIRouter, HTTPException, Body
import inspect
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
from app.services.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    wave2vec,
    extract_whisper_embeddings,
    extract_wav2vec2_embeddings,
    reduce_dimensions,
)
from app.services.dataset_service import resolve_file
from app.core.redis import get_result, cache_result

router = APIRouter()
logger = logging.getLogger(__name__)


MODEL_FUNCTIONS = {
    "whisper-base": transcribe_whisper_base,
    "whisper-large": transcribe_whisper_large,
    "wav2vec2": wave2vec,
}


@router.post("/inferences/run")
async def run_inference_endpoint(
    request: dict = Body(..., example={
        "model": "whisper-base",
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice", 
        "dataset_file": "sample-001.mp3"
    })
):
    # Extract parameters from request body
    model = request.get("model")
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    
    return await run_inference(model, file_path, dataset, dataset_file)


@router.post("/inferences/batch-check")
async def check_batch_cache(
    request: dict = Body(..., example={
        "model": "whisper-base",
        "dataset": "common-voice",
        "files": ["sample-001.mp3", "sample-002.mp3"]
    })
):
    """Check which files in a batch already have cached predictions"""
    model = request.get("model")
    dataset = request.get("dataset") 
    files = request.get("files", [])
    
    if not model or not dataset:
        raise HTTPException(status_code=400, detail="Model and dataset are required")
    
    cached_results = {}
    missing_files = []
    
    for filename in files:
        try:
            # Resolve the file path
            resolved_path = resolve_file(dataset, filename)
            
            # Create cache key
            file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
            cache_key = f"{model}_{file_content_hash}"
            
            # Check cache
            cached_result = await get_result(model, cache_key)
            if cached_result is not None:
                cached_results[filename] = cached_result.get("prediction", cached_result)
            else:
                missing_files.append(filename)
                
        except (FileNotFoundError, ValueError):
            # File doesn't exist or invalid dataset
            missing_files.append(filename)
    
    return {
        "cached_results": cached_results,
        "missing_files": missing_files,
        "cache_hit_rate": len(cached_results) / len(files) if files else 0
    }


async def run_inference(
    model: str,
    file_path: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_file: Optional[str] = None,
):
    """Internal function for running inference - can be called directly or via HTTP endpoint"""
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

    # Create cache key based on model and file path
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"{model}_{file_content_hash}"
    
    # Check if result is cached
    cached_result = await get_result(model, cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached result for {resolved_path}")
        return cached_result.get("prediction", cached_result)

    # Detect if function is async or sync and call appropriately
    if inspect.iscoroutinefunction(func):
        prediction = await func(str(resolved_path))
    else:
        prediction = await asyncio.to_thread(func, str(resolved_path))

    # Cache the result for future use (6 hours TTL)
    await cache_result(model, cache_key, {"prediction": prediction}, ttl=6*60*60)
    logger.info(f"Cached prediction for {resolved_path}")

    return prediction


@router.post("/inferences/embeddings")
async def extract_embeddings_endpoint(
    request: dict = Body(..., example={
        "model": "whisper-base",
        "dataset": "common-voice",
        "files": ["sample-001.mp3", "sample-002.mp3"],
        "reduction_method": "pca",
        "n_components": 3
    })
):
    """Extract embeddings from multiple audio files and optionally reduce dimensions"""
    model = request.get("model")
    dataset = request.get("dataset")
    files = request.get("files", [])
    reduction_method = request.get("reduction_method", "pca")
    n_components = request.get("n_components", 3)
    
    if not model or not dataset or not files:
        raise HTTPException(status_code=400, detail="Model, dataset, and files are required")
    
    logger.info(f"Extracting embeddings for {len(files)} files with model {model}")
    
    embeddings_data = []
    embeddings_list = []
    
    for filename in files:
        try:
            # Resolve the file path
            resolved_path = resolve_file(dataset, filename)
            
            # Create cache key for embeddings
            file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
            cache_key = f"{model}_embeddings_{file_content_hash}"
            
            # Check if embeddings are cached
            cached_embeddings = await get_result(model, cache_key)
            
            if cached_embeddings is not None:
                embedding = cached_embeddings.get("embedding")
                logger.info(f"Using cached embeddings for {filename}")
            else:
                # Extract embeddings based on model type
                if model.startswith("whisper"):
                    model_size = "base" if "base" in model else "large"
                    embedding = await asyncio.to_thread(extract_whisper_embeddings, str(resolved_path), model_size)
                elif model == "wav2vec2":
                    embedding = await asyncio.to_thread(extract_wav2vec2_embeddings, str(resolved_path))
                else:
                    raise HTTPException(status_code=400, detail=f"Embedding extraction not supported for model: {model}")
                
                # Cache the embeddings (24 hours TTL since embeddings don't change)
                await cache_result(model, cache_key, {"embedding": embedding.tolist()}, ttl=24*60*60)
                logger.info(f"Cached embeddings for {filename}")
            
            # Convert back to numpy array if it was cached as list
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            embeddings_data.append({
                "filename": filename,
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding)
            })
            embeddings_list.append(embedding)
            
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {filename}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Error extracting embeddings for {filename}: {str(e)}")
            continue
    
    if not embeddings_list:
        raise HTTPException(status_code=400, detail="No valid embeddings could be extracted")
    
    # Perform dimensionality reduction if requested
    reduced_embeddings = None
    if reduction_method and len(embeddings_list) > 1:
        try:
            reduced_embeddings = await asyncio.to_thread(
                reduce_dimensions, embeddings_list, reduction_method, n_components
            )
            logger.info(f"Reduced {len(embeddings_list)} embeddings from {embeddings_list[0].shape[0]}D to {n_components}D using {reduction_method}")
        except Exception as e:
            logger.warning(f"Dimensionality reduction failed: {str(e)}")
    
    # Prepare response
    response = {
        "model": model,
        "dataset": dataset,
        "reduction_method": reduction_method,
        "n_components": n_components,
        "embeddings": embeddings_data,
        "total_files": len(embeddings_data),
        "original_dimension": embeddings_list[0].shape[0] if embeddings_list else 0
    }
    
    if reduced_embeddings is not None:
        response["reduced_embeddings"] = [
            {
                "filename": embeddings_data[i]["filename"],
                "coordinates": reduced_embeddings[i].tolist()
            }
            for i in range(len(reduced_embeddings))
        ]
    
    return response


@router.post("/inferences/embeddings/single")
async def extract_single_embedding_endpoint(
    request: dict = Body(..., example={
        "model": "whisper-base",
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice",
        "dataset_file": "sample-001.mp3"
    })
):
    """Extract embeddings from a single audio file"""
    model = request.get("model")
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    
    # Resolve file path
    resolved_path = None
    if file_path:
        resolved_path = Path(file_path)
    elif dataset and dataset_file:
        try:
            resolved_path = resolve_file(dataset, dataset_file)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing audio reference. Provide either 'file_path' or 'dataset' + 'dataset_file'."
        )
    
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {resolved_path}")
    
    # Create cache key for embeddings
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"{model}_embeddings_{file_content_hash}"
    
    # Check if embeddings are cached
    cached_embeddings = await get_result(model, cache_key)
    
    if cached_embeddings is not None:
        embedding = cached_embeddings.get("embedding")
        logger.info(f"Using cached embeddings for {resolved_path}")
    else:
        # Extract embeddings based on model type
        if model.startswith("whisper"):
            model_size = "base" if "base" in model else "large"
            embedding = await asyncio.to_thread(extract_whisper_embeddings, str(resolved_path), model_size)
        elif model == "wav2vec2":
            embedding = await asyncio.to_thread(extract_wav2vec2_embeddings, str(resolved_path))
        else:
            raise HTTPException(status_code=400, detail=f"Embedding extraction not supported for model: {model}")
        
        # Cache the embeddings (24 hours TTL)
        await cache_result(model, cache_key, {"embedding": embedding.tolist()}, ttl=24*60*60)
        logger.info(f"Cached embeddings for {resolved_path}")
    
    # Convert back to numpy array if it was cached as list
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    return {
        "model": model,
        "file_path": str(resolved_path),
        "embedding": embedding.tolist(),
        "embedding_dim": len(embedding)
    }
