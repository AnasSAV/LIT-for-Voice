from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from app.services.pertubation_service import perturb_and_save

router = APIRouter()
logger = logging.getLogger(__name__)


def get_session_id(request: Request) -> Optional[str]:
    """Extract session ID from request (optional for backwards compatibility)"""
    return getattr(request.state, 'sid', None)

class Perturbation(BaseModel):
    type: str
    params: Dict[str, Any] = {}

class PerturbRequest(BaseModel):
    file_path: str
    perturbations: List[Perturbation]
    dataset: Optional[str] = None

# --- Response schema ---
class PerturbResponse(BaseModel):
    perturbed_file: str
    filename: str
    duration_ms: int
    sample_rate: int
    applied_perturbations: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None

# --- Endpoint ---
@router.post("/perturb", response_model=PerturbResponse)
def apply_perturbations(http_request: Request, request: PerturbRequest):
    """
    Apply multiple perturbations to the given audio file and save the result.
    """
    try:
        session_id = get_session_id(http_request)
        
        # Convert Pydantic models to dictionaries for the service
        perturbations_list = []
        for perturbation in request.perturbations:
            perturbations_list.append({
                "type": perturbation.type,
                "params": perturbation.params
            })
        
        # Apply perturbations and save (file validation happens inside the service)
        result = perturb_and_save(
            file_path=request.file_path,
            perturbations=perturbations_list,
            output_dir="uploads",
            dataset=request.dataset,
            session_id=session_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        
        logger.info(f"Successfully applied perturbations to {request.file_path}")
        
        return PerturbResponse(
            perturbed_file=result["perturbed_file"],
            filename=result["filename"],
            duration_ms=result["duration_ms"],
            sample_rate=result["sample_rate"],
            applied_perturbations=result["applied_perturbations"],
            success=result["success"]
        )
        
    except Exception as e:
        logger.error(f"Error applying perturbations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply perturbations: {str(e)}")