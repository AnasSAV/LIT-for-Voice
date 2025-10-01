from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/debug/session")
async def get_session_info(request: Request):
    """Debug endpoint to see current session information"""
    session_id = getattr(request.state, 'sid', None)
    cookies = dict(request.cookies)
    
    return JSONResponse({
        "session_id": session_id,
        "cookies": cookies,
        "headers": dict(request.headers)
    })