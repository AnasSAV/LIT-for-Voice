from pathlib import Path
from typing import Dict, List, Optional
import csv
import librosa
import logging
from .custom_dataset_service import (
    get_custom_dataset_manager, 
    is_custom_dataset, 
    parse_custom_dataset_name
)

logger = logging.getLogger(__name__)


# Resolve paths relative to repo structure: Backend/data/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# CSV metadata files per dataset
DATASET_PATHS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv",
    "ravdess": DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv",
}

# Base directories for dataset audio files
DATASET_BASE_DIRS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev",
    "ravdess": DATA_DIR / "ravdess_subset",
}


def calculate_audio_duration(audio_path: Path) -> float:
    """Calculate duration of audio file in seconds"""
    try:
        # Use librosa to get duration without loading the entire audio
        duration = librosa.get_duration(path=str(audio_path))
        return round(duration, 2)
    except Exception as e:
        logger.warning(f"Could not calculate duration for {audio_path}: {e}")
        return 0.0

def load_metadata(dataset: str, session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Load metadata for both global and custom datasets"""
    
    # Handle custom datasets
    if is_custom_dataset(dataset):
        if not session_id:
            raise ValueError("session_id is required for custom datasets")
        
        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
        if session_id_from_name != session_id:
            raise ValueError("Session ID mismatch for custom dataset")
        
        manager = get_custom_dataset_manager(session_id)
        return manager.get_dataset_files_as_csv_format(dataset_name)
    
    # Handle global datasets (existing logic)
    ds = dataset.lower()
    if ds not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}")

    csv_path = DATASET_PATHS[ds]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found for: {dataset}")

    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys to lowercase; strip whitespace
                normalized = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                
                # Add duration for datasets that don't have it (like RAVDESS)
                if ds == "ravdess" and "duration" not in normalized:
                    try:
                        # Try to find the audio file and calculate duration
                        filename = normalized.get("filename", "")
                        if filename:
                            audio_path = DATASET_BASE_DIRS[ds] / filename
                            if audio_path.exists():
                                duration = calculate_audio_duration(audio_path)
                                normalized["duration"] = str(duration)
                            else:
                                normalized["duration"] = "0.0"
                        else:
                            normalized["duration"] = "0.0"
                    except Exception as e:
                        logger.warning(f"Error calculating duration for {filename}: {e}")
                        normalized["duration"] = "0.0"
                
                rows.append(normalized)
    except Exception:
        # Re-raise to let the route map to a 500
        raise

    return rows


def resolve_file(dataset: str, file_path: str, session_id: Optional[str] = None) -> Path:
    """Resolve file path for both global and custom datasets"""
    
    # Handle custom datasets
    if is_custom_dataset(dataset):
        if not session_id:
            raise ValueError("session_id is required for custom datasets")
        
        session_id_from_name, dataset_name = parse_custom_dataset_name(dataset)
        if session_id_from_name != session_id:
            raise ValueError("Session ID mismatch for custom dataset")
        
        manager = get_custom_dataset_manager(session_id)
        return manager.resolve_file_path(dataset_name, file_path)
    
    # Handle global datasets (existing logic)
    ds = dataset.lower()
    if ds not in DATASET_BASE_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}")

    base_dir = DATASET_BASE_DIRS[ds]
    safe_name = Path(file_path).name
    audio_path = base_dir / safe_name
    if not audio_path.exists() or not audio_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {safe_name}")

    return audio_path


def media_type_for(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    media_type_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
    }
    media_type = media_type_map.get(ext)
    if media_type is None:
        raise ValueError(f"Unsupported media type for extension: {ext}")
    return media_type
