from pathlib import Path
from typing import Dict, List
import csv


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


def load_metadata(dataset: str) -> List[Dict[str, str]]:

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
                rows.append(normalized)
    except Exception:
        # Re-raise to let the route map to a 500
        raise

    return rows


def resolve_file(dataset: str, file_path: str) -> Path:

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
