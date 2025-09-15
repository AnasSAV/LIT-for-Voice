from fastapi import APIRouter, HTTPException, Body, Request
import inspect
import asyncio
import logging
import hashlib
import difflib
import re
import string
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
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


@router.post("/inferences/whisper-accuracy")
async def get_whisper_accuracy(request: Request):
    """
    Get whisper prediction from cache and compare with ground truth
    """
    try:
        body = await request.json()
        model = body.get("model", "whisper-base")
        dataset = body.get("dataset")
        dataset_file = body.get("dataset_file")
        file_path = body.get("file_path")
        
        if not dataset_file and not file_path:
            raise HTTPException(status_code=400, detail="Either dataset_file or file_path must be provided")
        
        # Get file path
        if file_path:
            resolved_path = Path(file_path)
        else:
            resolved_path = resolve_file(dataset, dataset_file)
        
        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Create cache key and get cached prediction
        file_content_hash = hashlib.md5(str(resolved_path).encode()).hexdigest()
        cache_key = f"{model}_{file_content_hash}"
        
        cached_result = await get_result(model, cache_key)
        
        print(f"DEBUG: Looking for cache key: {cache_key}")
        print(f"DEBUG: Cached result found: {cached_result is not None}")
        
        if cached_result is None:
            # If not cached, run inference first
            print(f"DEBUG: No cached result found, running inference for {dataset_file}")
            try:
                # Run inference to get the prediction and cache it
                if dataset and dataset_file:
                    inference_result = await run_inference(model, None, dataset, dataset_file)
                else:
                    inference_result = await run_inference(model, str(resolved_path), None, None)
                
                # Now try to get the cached result again
                cached_result = await get_result(model, cache_key)
                if cached_result is None:
                    # If still not cached, use the inference result directly
                    cached_result = {"prediction": inference_result}
            except Exception as e:
                print(f"DEBUG: Failed to run inference: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to run inference for {dataset_file}: {str(e)}")
        
        # Extract transcript from cached result
        if isinstance(cached_result, dict):
            predicted_transcript = cached_result.get("prediction", cached_result.get("transcript", ""))
        else:
            predicted_transcript = str(cached_result)
        
        # Get ground truth from dataset metadata
        ground_truth = ""
        if dataset and dataset_file:
            # Load dataset metadata
            if dataset == "common-voice":
                metadata_path = DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv"
            elif dataset == "ravdess":
                metadata_path = DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv"
            else:
                raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")
            
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                # Find the row for this file
                # Try both with and without path prefix
                file_rows = df[df['filename'] == dataset_file]
                if file_rows.empty:
                    # Try with path prefix for common-voice
                    if dataset == "common-voice":
                        file_rows = df[df['filename'] == f"cv-valid-dev/{dataset_file}"]
                
                if not file_rows.empty:
                    # Try different column names for transcript/text
                    if dataset == "common-voice":
                        # For common-voice, use 'text' column
                        for col in ['text', 'transcript', 'sentence', 'label']:
                            if col in df.columns:
                                ground_truth = str(file_rows.iloc[0][col])
                                break
                    elif dataset == "ravdess":
                        # For RAVDESS, use 'statement' column
                        for col in ['statement', 'text', 'transcript', 'sentence']:
                            if col in df.columns:
                                ground_truth = str(file_rows.iloc[0][col])
                                break
        
        if not ground_truth:
            # More specific error message for debugging
            error_msg = f"Ground truth not found in dataset metadata. Dataset: {dataset}, File: {dataset_file}"
            if dataset and dataset_file:
                if dataset == "common-voice":
                    metadata_path = DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv"
                elif dataset == "ravdess":
                    metadata_path = DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv"
                else:
                    metadata_path = None
                if metadata_path:
                    error_msg += f", Metadata path: {metadata_path}, Exists: {metadata_path.exists()}"
            print(f"DEBUG: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Calculate accuracy metrics
        def calculate_accuracy(predicted, ground_truth):
            # Clean and normalize both strings
            def clean_text(text):
                # Convert to lowercase
                text = text.lower()
                # Remove punctuation
                text = text.translate(str.maketrans('', '', string.punctuation))
                # Remove extra whitespace and normalize
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            pred_clean = clean_text(predicted)
            truth_clean = clean_text(ground_truth)
            
            print(f"DEBUG: Predicted clean: '{pred_clean}'")
            print(f"DEBUG: Truth clean: '{truth_clean}'")
            
            # Split into words for word-based metrics
            pred_words = pred_clean.split()
            truth_words = truth_clean.split()
            
            # Character-based similarity (after cleaning)
            char_similarity = difflib.SequenceMatcher(None, pred_clean, truth_clean).ratio()
            
            # Word-based similarity (after cleaning)
            word_similarity = difflib.SequenceMatcher(None, pred_words, truth_words).ratio()
            
            # Exact match accuracy
            exact_match = 1.0 if pred_clean == truth_clean else 0.0
            
            # Levenshtein distance (character-level)
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            lev_dist = levenshtein_distance(pred_clean, truth_clean)
            
            # Word Error Rate (WER) - standard ASR metric
            def calculate_wer(predicted_words, reference_words):
                # This is a simplified WER calculation using edit distance on word level
                if len(reference_words) == 0:
                    return 1.0 if len(predicted_words) > 0 else 0.0
                
                # Calculate word-level edit distance
                word_lev_dist = levenshtein_distance(predicted_words, reference_words)
                wer = word_lev_dist / len(reference_words)
                return min(wer, 1.0)  # Cap at 1.0
            
            # Character Error Rate (CER)
            def calculate_cer(predicted_chars, reference_chars):
                if len(reference_chars) == 0:
                    return 1.0 if len(predicted_chars) > 0 else 0.0
                
                char_lev_dist = levenshtein_distance(predicted_chars, reference_chars)
                cer = char_lev_dist / len(reference_chars)
                return min(cer, 1.0)  # Cap at 1.0
            
            wer = calculate_wer(pred_words, truth_words)
            cer = calculate_cer(pred_clean, truth_clean)
            
            # Overall accuracy based on word similarity (most intuitive)
            accuracy_percentage = word_similarity * 100
            
            return {
                "accuracy_percentage": accuracy_percentage,
                "word_error_rate": wer,
                "character_error_rate": cer,
                "levenshtein_distance": lev_dist,
                "exact_match": exact_match,
                "character_similarity": char_similarity * 100,
                "word_count_predicted": len(pred_words),
                "word_count_truth": len(truth_words)
            }
        
        accuracy_metrics = calculate_accuracy(predicted_transcript, ground_truth)
        
        return {
            "predicted_transcript": predicted_transcript,
            "ground_truth": ground_truth,
            **accuracy_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in whisper accuracy calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Accuracy calculation failed: {str(e)}")


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
