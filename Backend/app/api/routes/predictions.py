from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import csv
import logging
from typing import Dict, List, Optional
from app.services.model_loader_service import (
    transcribe_whisper_base,
    wave2vec,
    predict_emotion_wave2vec
)

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = logging.getLogger(__name__)

# Resolve paths relative to repo structure: Backend/data/
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAVDESS_DIR = DATA_DIR / "ravdess_subset"
COMMON_VOICE_DIR = DATA_DIR / "common_voice_valid_dev"

@router.post("/batch")
async def run_batch_predictions(
    model: str = Query(..., description="Model to use for predictions"),
    dataset: str = Query(..., description="Dataset to run predictions on"),
    limit: Optional[int] = Query(None, description="Limit number of files to process")
):
    """
    Run batch predictions on a dataset
    """
    if model not in ["whisper-base", "whisper-large", "wav2vec2"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    
    if dataset not in ["ravdess", "common-voice"]:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
    
    try:
        # Get dataset metadata
        if dataset == "ravdess":
            metadata_file = RAVDESS_DIR / "ravdess_subset_metadata.csv"
            audio_dir = RAVDESS_DIR
        else:
            metadata_file = COMMON_VOICE_DIR / "common_voice_valid_data_metadata.csv"
            audio_dir = COMMON_VOICE_DIR
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail=f"Metadata file not found: {metadata_file}")
        
        # Read metadata
        predictions = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Limit processing if specified
            if limit:
                rows = rows[:limit]
            
            for i, row in enumerate(rows):
                try:
                    filename = row.get('filename', '')
                    if not filename:
                        continue
                        
                    audio_path = audio_dir / filename
                    if not audio_path.exists():
                        logger.warning(f"Audio file not found: {audio_path}")
                        continue
                    
                    # Run prediction based on model
                    if model in ["whisper-base", "whisper-large"]:
                        if model == "whisper-base":
                            transcript = transcribe_whisper_base(str(audio_path))
                        else:
                            # Add whisper-large implementation if needed
                            transcript = transcribe_whisper_base(str(audio_path))
                        
                        prediction_result = {
                            "filename": filename,
                            "prediction_type": "transcript",
                            "prediction": transcript,
                            "ground_truth": row.get('sentence', '') if dataset == "common-voice" else row.get('statement', ''),
                            "metadata": row
                        }
                    else:  # wav2vec2
                        emotion = predict_emotion_wave2vec(str(audio_path))
                        
                        # For wav2vec2, also get transcript using whisper-base
                        transcript = transcribe_whisper_base(str(audio_path))
                        
                        prediction_result = {
                            "filename": filename,
                            "prediction_type": "emotion",
                            "emotion_prediction": emotion,
                            "transcript": transcript,
                            "ground_truth_emotion": row.get('emotion', ''),
                            "ground_truth_transcript": row.get('statement', ''),
                            "metadata": row
                        }
                    
                    predictions.append(prediction_result)
                    logger.info(f"Processed {i+1}/{len(rows)}: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    continue
        
        return JSONResponse(
            status_code=200,
            content={
                "model": model,
                "dataset": dataset,
                "total_files": len(rows),
                "processed_files": len(predictions),
                "predictions": predictions
            }
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/single")
async def run_single_prediction(
    model: str = Query(..., description="Model to use for predictions"),
    dataset: str = Query(..., description="Dataset the file belongs to"),
    filename: str = Query(..., description="Filename to run prediction on")
):
    """
    Run prediction on a single dataset file
    """
    if model not in ["whisper-base", "whisper-large", "wav2vec2"]:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    
    if dataset not in ["ravdess", "common-voice"]:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
    
    try:
        # Get dataset paths
        if dataset == "ravdess":
            metadata_file = RAVDESS_DIR / "ravdess_subset_metadata.csv"
            audio_dir = RAVDESS_DIR
        else:
            metadata_file = COMMON_VOICE_DIR / "common_voice_valid_data_metadata.csv"
            audio_dir = COMMON_VOICE_DIR
        
        audio_path = audio_dir / filename
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")
        
        # Get metadata for this file
        metadata_row = None
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('filename') == filename:
                    metadata_row = row
                    break
        
        if not metadata_row:
            raise HTTPException(status_code=404, detail=f"Metadata not found for file: {filename}")
        
        # Run prediction based on model
        if model in ["whisper-base", "whisper-large"]:
            if model == "whisper-base":
                transcript = transcribe_whisper_base(str(audio_path))
            else:
                # Add whisper-large implementation if needed
                transcript = transcribe_whisper_base(str(audio_path))
            
            result = {
                "filename": filename,
                "prediction_type": "transcript",
                "prediction": transcript,
                "ground_truth": metadata_row.get('sentence', '') if dataset == "common-voice" else metadata_row.get('statement', ''),
                "metadata": metadata_row
            }
        else:  # wav2vec2
            emotion = predict_emotion_wave2vec(str(audio_path))
            
            # For wav2vec2, also get transcript using whisper-base
            transcript = transcribe_whisper_base(str(audio_path))
            
            result = {
                "filename": filename,
                "prediction_type": "emotion",
                "emotion_prediction": emotion,
                "transcript": transcript,
                "ground_truth_emotion": metadata_row.get('emotion', ''),
                "ground_truth_transcript": metadata_row.get('statement', ''),
                "metadata": metadata_row
            }
        
        return JSONResponse(status_code=200, content=result)
        
    except Exception as e:
        logger.error(f"Single prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Single prediction failed: {str(e)}")

@router.get("/dataset/{dataset}/files")
async def get_dataset_files(
    dataset: str,
    limit: Optional[int] = Query(None, description="Limit number of files returned")
):
    """
    Get list of audio files in a dataset with their metadata
    """
    if dataset not in ["ravdess", "common-voice"]:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
    
    try:
        if dataset == "ravdess":
            metadata_file = RAVDESS_DIR / "ravdess_subset_metadata.csv"
            audio_dir = RAVDESS_DIR
        else:
            metadata_file = COMMON_VOICE_DIR / "common_voice_valid_data_metadata.csv"
            audio_dir = COMMON_VOICE_DIR
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail=f"Metadata file not found for dataset: {dataset}")
        
        files = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Limit if specified
            if limit:
                rows = rows[:limit]
            
            for row in rows:
                filename = row.get('filename', '')
                if not filename:
                    continue
                    
                audio_path = audio_dir / filename
                if audio_path.exists():
                    files.append({
                        "filename": filename,
                        "exists": True,
                        "metadata": row
                    })
                else:
                    files.append({
                        "filename": filename,
                        "exists": False,
                        "metadata": row
                    })
        
        return JSONResponse(
            status_code=200,
            content={
                "dataset": dataset,
                "total_files": len(files),
                "files": files
            }
        )
        
    except Exception as e:
        logger.error(f"Get dataset files error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset files: {str(e)}")
