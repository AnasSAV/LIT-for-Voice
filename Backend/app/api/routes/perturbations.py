from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

class Perturbation(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class PerturbRequest(BaseModel):
    file_path: str
    perturbations: List[Perturbation]

# --- Response schema ---
class PerturbResponse(BaseModel):
    audio_url: str
    duration_ms: int
    metrics: Optional[Dict[str, float]] = None
    applied: List[Perturbation]

# --- Endpoint ---
@router.post("/perturb", response_model=PerturbResponse)
def apply_perturbations(request: PerturbRequest):
    """
    Apply multiple perturbations to the given audio in one request.
    """
    # Here you’d load audio by request.audio_id + session_id
    # Apply perturbations in order
    # Save perturbed audio & return URL + metrics
    
    return PerturbResponse(
        audio_url=f"https://server/sessions/{request.session_id}/audio/{request.audio_id}_perturbed.wav",
        duration_ms=45000,
        metrics={"wer_delta": 0.05, "conf_delta": -0.02},
        applied=request.perturbations
    )