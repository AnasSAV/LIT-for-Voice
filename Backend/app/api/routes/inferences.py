from fastapi import APIRouter, HTTPException, Body, Request
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
    predict_emotion_wave2vec,
)
from app.services.dataset_service import resolve_file
from app.core.redis import get_result, cache_result

router = APIRouter()

# Define paths
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
UPLOAD_DIR = Path("uploads")

# Dataset directories
DATASET_DIRS = {
    "common-voice": DATA_DIR / "common_voice_valid_dev",
    "ravdess": DATA_DIR / "ravdess_subset",
}
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


@router.post("/inferences/whisper-batch")
async def batch_whisper_analysis(request: Request):
    """
    Get batch whisper transcripts from cache and analyze common terms
    """
    try:
        body = await request.json()
        filenames = body.get("filenames", [])
        model = body.get("model", "whisper-base")
        dataset = body.get("dataset")
        
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        if len(filenames) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per batch.")
        
        # Process each file - try to get from cache first
        individual_transcripts = []
        all_words = []
        cached_count = 0
        missing_count = 0
        
        for filename in filenames:
            try:
                # Get file path and create cache key
                if dataset:
                    file_path = resolve_file(dataset, filename)
                else:
                    file_path = UPLOAD_DIR / filename
                    if not file_path.exists():
                        print(f"Warning: File not found: {file_path}")
                        missing_count += 1
                        continue
                
                # Create cache key (same as used in regular inference)
                file_content_hash = hashlib.md5(str(file_path).encode()).hexdigest()
                cache_key = f"{model}_{file_content_hash}"
                
                # Try to get from cache first
                cached_result = await get_result(model, cache_key)
                
                transcript = None
                if cached_result is not None:
                    # Extract transcript from cached result
                    if isinstance(cached_result, dict):
                        transcript = cached_result.get("prediction", cached_result.get("transcript"))
                    else:
                        transcript = cached_result
                    cached_count += 1
                else:
                    # Not in cache - skip this file for now
                    print(f"No cached transcript found for {filename}")
                    missing_count += 1
                    continue
                
                if transcript:
                    # Clean and tokenize transcript
                    words = transcript.lower().split()
                    # Remove common stop words and punctuation
                    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
                    filtered_words = [word.strip('.,!?";:()[]{}').lower() for word in words if word.strip('.,!?";:()[]{}').lower() not in stop_words and len(word.strip('.,!?";:()[]{}')) > 2]
                    
                    individual_transcripts.append({
                        "filename": filename,
                        "transcript": transcript,
                        "word_count": len(words)
                    })
                    
                    all_words.extend(filtered_words)
                    
            except Exception as file_error:
                print(f"Error processing {filename}: {file_error}")
                missing_count += 1
                continue
        
        if not individual_transcripts:
            raise HTTPException(status_code=404, detail=f"No cached transcripts found for the selected files. Found {cached_count} cached, {missing_count} missing. Please run inference on these files first.")
        
        # Calculate word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Get top terms with percentages
        common_terms = []
        for word, count in word_counts.most_common(10):  # Get top 10, frontend will show top 5
            percentage = (count / total_words) * 100
            common_terms.append({
                "term": word,
                "count": count,
                "percentage": percentage
            })
        
        return {
            "common_terms": common_terms,
            "individual_transcripts": individual_transcripts,
            "summary": {
                "total_files": len(individual_transcripts),
                "total_words": total_words,
                "unique_words": len(word_counts),
                "avg_words_per_file": sum(t["word_count"] for t in individual_transcripts) / len(individual_transcripts)
            },
            "cache_info": {
                "cached_count": cached_count,
                "missing_count": missing_count,
                "cache_hit_rate": cached_count / len(filenames) if filenames else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in batch whisper analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.post("/inferences/wav2vec2-batch")
async def batch_wav2vec2_prediction(request: Request):
    """
    Get batch wav2vec2 predictions for multiple files and calculate aggregated probabilities
    """
    try:
        body = await request.json()
        filenames = body.get("filenames", [])
        dataset = body.get("dataset")
        
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        if len(filenames) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Too many files. Maximum 50 files per batch.")
        
        # Process each file
        individual_predictions = []
        predicted_emotions = []  # Store just the predicted emotions for distribution
        
        for filename in filenames:
            try:
                # Load and predict for each file
                if dataset:
                    file_path = resolve_file(dataset, filename)
                else:
                    file_path = UPLOAD_DIR / filename
                    if not file_path.exists():
                        print(f"Warning: File not found: {file_path}")
                        continue
                
                # Get detailed prediction
                result = predict_emotion_wave2vec(str(file_path))
                
                individual_predictions.append({
                    "filename": filename,
                    "predicted_emotion": result["predicted_emotion"],
                    "probabilities": result["probabilities"], 
                    "confidence": result["confidence"]
                })
                
                # Store the predicted emotion for distribution calculation
                predicted_emotions.append(result["predicted_emotion"])
                    
            except Exception as file_error:
                print(f"Error processing {filename}: {file_error}")
                continue
        
        if not individual_predictions:
            raise HTTPException(status_code=404, detail="No valid files could be processed")
        
        # Calculate emotion distribution (percentage of files predicted as each emotion)
        emotion_counts = {}
        for emotion in predicted_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_files = len(predicted_emotions)
        emotion_distribution = {}
        for emotion, count in emotion_counts.items():
            emotion_distribution[emotion] = count / total_files
        
        # Find dominant emotion (most frequent prediction)
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            "emotion_distribution": emotion_distribution,  # Percentage of files predicted as each emotion
            "emotion_counts": emotion_counts,  # Raw counts
            "individual_predictions": individual_predictions,
            "summary": {
                "total_files": total_files,
                "dominant_emotion": dominant_emotion[0],
                "dominant_count": dominant_emotion[1],
                "dominant_percentage": dominant_emotion[1] / total_files
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in batch wav2vec2 prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/inferences/wav2vec2-detailed")
async def get_wav2vec2_detailed_prediction(
    request: dict = Body(..., example={
        "file_path": "/path/to/audio.wav",
        "dataset": "common-voice", 
        "dataset_file": "sample-001.mp3"
    })
):
    """Get detailed wav2vec2 prediction with probabilities for all emotions"""
    file_path = request.get("file_path")
    dataset = request.get("dataset")
    dataset_file = request.get("dataset_file")
    
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
    
    # Create cache key for detailed predictions
    file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
    cache_key = f"wav2vec2_detailed_{file_content_hash}"
    
    # Check if result is cached
    cached_result = await get_result("wav2vec2", cache_key)
    if cached_result is not None:
        logger.info(f"Returning cached detailed wav2vec2 result for {resolved_path}")
        return cached_result.get("prediction", cached_result)
    
    # Get detailed prediction with probabilities
    try:
        detailed_result = await asyncio.to_thread(predict_emotion_wave2vec, str(resolved_path))
        
        # Cache the detailed result
        await cache_result("wav2vec2", cache_key, {"prediction": detailed_result}, ttl=6*60*60)
        logger.info(f"Cached detailed wav2vec2 prediction for {resolved_path}")
        
        return detailed_result
        
    except Exception as e:
        logger.error(f"Error getting detailed wav2vec2 prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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
    
    logger.info(f"Successfully extracted embeddings for {len(embeddings_list)} files")
    
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
            # Return error details for debugging
            response_error = {
                "error": f"Dimensionality reduction failed: {str(e)}",
                "embeddings_count": len(embeddings_list),
                "embedding_dimension": embeddings_list[0].shape[0] if embeddings_list else 0
            }
            raise HTTPException(status_code=500, detail=response_error)
    
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
