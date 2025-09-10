from fastapi import APIRouter, HTTPException, Response
from typing import List, Dict, Optional
import logging

from app.services.prediction_cache import prediction_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/database", tags=["database"])

@router.get("/stats")
async def get_database_stats():
    """Get comprehensive database and cache statistics"""
    try:
        stats = await prediction_cache.get_cache_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions")
async def list_predictions(model: Optional[str] = None, dataset: Optional[str] = None):
    """List all cached predictions, optionally filtered by model and/or dataset"""
    try:
        predictions = await prediction_cache.list_cached_predictions()
        
        # Apply filters if provided
        if model:
            predictions = [p for p in predictions if p.get('model') == model]
        if dataset:
            predictions = [p for p in predictions if p.get('dataset') == dataset]
        
        return {
            "status": "success",
            "predictions": predictions,
            "total": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error listing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/predictions")
async def clear_predictions(model: Optional[str] = None, dataset: Optional[str] = None):
    """Clear cached predictions, optionally filtered by model and/or dataset"""
    try:
        deleted_count = await prediction_cache.invalidate(model=model, dataset=dataset)
        return {
            "status": "success",
            "message": f"Cleared {deleted_count} prediction records",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_database(output_filename: Optional[str] = None):
    """Export database to SQL file"""
    try:
        exported_file = await prediction_cache.export_database(output_filename)
        return {
            "status": "success",
            "message": "Database exported successfully",
            "exported_file": exported_file
        }
    except Exception as e:
        logger.error(f"Error exporting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/download")
async def download_database_export():
    """Download the latest database export as SQL file"""
    try:
        # Export to a temporary file
        exported_file = await prediction_cache.export_database()
        
        # Read the file content
        with open(exported_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Return as downloadable file
        return Response(
            content=content,
            media_type="application/sql",
            headers={
                "Content-Disposition": f"attachment; filename={exported_file}"
            }
        )
    except Exception as e:
        logger.error(f"Error downloading database export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebuild-cache")
async def rebuild_cache_from_database():
    """Rebuild Redis and file caches from database"""
    try:
        # Get all predictions from database
        predictions_list = await prediction_cache.list_cached_predictions()
        rebuilt_count = 0
        
        # For each prediction, load from database and store in caches
        for pred_info in predictions_list:
            try:
                # Get full prediction data from database
                from app.services.database import prediction_db
                db_data = await prediction_db.get_predictions(
                    pred_info['model'], 
                    pred_info['dataset']
                )
                
                if db_data:
                    # Store back in caches (this will populate Redis and file cache)
                    cache_key = prediction_cache._generate_cache_key(
                        pred_info['model'], 
                        pred_info['dataset']
                    )
                    await prediction_cache._restore_to_caches(
                        cache_key, 
                        pred_info['model'], 
                        pred_info['dataset'], 
                        db_data
                    )
                    rebuilt_count += 1
                    
            except Exception as e:
                logger.error(f"Error rebuilding cache for {pred_info['model']}-{pred_info['dataset']}: {e}")
        
        return {
            "status": "success",
            "message": f"Rebuilt cache for {rebuilt_count} prediction sets",
            "rebuilt_count": rebuilt_count
        }
    except Exception as e:
        logger.error(f"Error rebuilding cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
