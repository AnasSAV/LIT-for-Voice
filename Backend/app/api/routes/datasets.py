from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict, List
import csv

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Resolve paths relative to repo structure: Backend/data/
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DATASET_PATHS: Dict[str, Path] = {
    "common-voice": DATA_DIR / "common_voice_valid_dev" / "common_voice_valid_data_metadata.csv",
    "ravdess": DATA_DIR / "ravdess_subset" / "ravdess_subset_metadata.csv",
}


@router.get("/{dataset}/metadata")
async def get_dataset_metadata(dataset: str) -> JSONResponse:
    ds = dataset.lower()
    if ds not in DATASET_PATHS:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")

    csv_path = DATASET_PATHS[ds]
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset metadata not found for: {dataset}")

    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys to lowercase; strip whitespace
                normalized = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                rows.append(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

    return JSONResponse(content=rows)
