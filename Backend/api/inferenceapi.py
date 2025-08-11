# apis/inferenceapi.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/run")
async def run_inference():
    return {"message": "This is inference endpoint"}
