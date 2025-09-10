import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiofiles
import asyncio
import redis.asyncio as redis
from app.core.redis import get_redis
from app.services.database import prediction_db

logger = logging.getLogger(__name__)

class PredictionCache:
    """
    Cache service for storing and retrieving prediction results using Redis and file backup
    """
    
    def __init__(self, cache_dir: str = "cache", cache_ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_ttl_seconds = int(cache_ttl_hours * 3600)
        self._redis_client: Optional[redis.Redis] = None
        # Initialize database
        asyncio.create_task(self._init_database())
        
    async def _init_database(self):
        """Initialize the database"""
        try:
            await prediction_db.initialize()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
        
    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client with error handling"""
        if self._redis_client is None:
            try:
                self._redis_client = await get_redis()
                # Test connection
                await self._redis_client.ping()
                logger.info("Redis connection established for caching")
            except Exception as e:
                logger.warning(f"Redis connection failed, falling back to file cache: {e}")
                self._redis_client = None
        return self._redis_client
    
    def _generate_cache_key(self, model: str, dataset: str, **kwargs) -> str:
        """Generate a unique cache key for model-dataset combination"""
        key_data = {
            "model": model,
            "dataset": dataset,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        return f"predictions:{cache_key}"
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        # Remove the 'predictions:' prefix for file naming
        file_key = cache_key.replace("predictions:", "")
        return self.cache_dir / f"{file_key}.json"
    
    async def get(self, model: str, dataset: str, **kwargs) -> Optional[Dict]:
        """
        Retrieve cached predictions for a model-dataset combination
        Priority: Redis -> File cache -> Database
        """
        cache_key = self._generate_cache_key(model, dataset, **kwargs)
        
        # Try Redis first
        redis_client = await self._get_redis()
        if redis_client:
            try:
                cached_data_str = await redis_client.get(cache_key)
                if cached_data_str:
                    cached_data = json.loads(cached_data_str)
                    logger.info(f"Cache hit (Redis) for {model}-{dataset}")
                    return cached_data["data"]
            except Exception as e:
                logger.error(f"Redis cache read error: {e}")
        
        # Try file cache
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    cached_data = json.loads(content)
                    
                if self._is_cache_valid(cached_data):
                    logger.info(f"Cache hit (file) for {model}-{dataset}")
                    
                    # Restore to Redis if available
                    if redis_client:
                        try:
                            await redis_client.setex(
                                cache_key, 
                                self.cache_ttl_seconds, 
                                json.dumps(cached_data)
                            )
                        except Exception as e:
                            logger.error(f"Redis cache write error during restore: {e}")
                    
                    return cached_data["data"]
                else:
                    # Remove expired cache file
                    cache_file.unlink()
                    logger.info(f"Expired cache removed for {model}-{dataset}")
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
                if cache_file.exists():
                    cache_file.unlink()
        
        # Try database as final fallback
        try:
            db_data = await prediction_db.get_predictions(model, dataset, **kwargs)
            if db_data:
                logger.info(f"Cache hit (database) for {model}-{dataset}")
                
                # Restore to faster caches
                await self._restore_to_caches(cache_key, model, dataset, db_data, **kwargs)
                
                return db_data
        except Exception as e:
            logger.error(f"Database cache read error: {e}")
        
        logger.info(f"Cache miss for {model}-{dataset}")
        return None
    
    async def _restore_to_caches(self, cache_key: str, model: str, dataset: str, data: Dict, **kwargs):
        """Restore data to Redis and file caches"""
        cached_data = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "dataset": dataset,
            "metadata": kwargs
        }
        
        cached_data_str = json.dumps(cached_data)
        
        # Restore to Redis
        redis_client = await self._get_redis()
        if redis_client:
            try:
                await redis_client.setex(cache_key, self.cache_ttl_seconds, cached_data_str)
            except Exception as e:
                logger.error(f"Redis restore error: {e}")
        
        # Restore to file cache
        cache_file = self._get_cache_file_path(cache_key)
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cached_data, indent=2))
        except Exception as e:
            logger.error(f"File cache restore error: {e}")
    
    async def set(self, model: str, dataset: str, data: Dict, **kwargs) -> None:
        """
        Store predictions in all cache layers: Redis, File, and Database
        """
        cache_key = self._generate_cache_key(model, dataset, **kwargs)
        
        cached_data = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "dataset": dataset,
            "metadata": kwargs
        }
        
        cached_data_str = json.dumps(cached_data)
        
        # Store in Redis first (fastest access)
        redis_client = await self._get_redis()
        if redis_client:
            try:
                await redis_client.setex(cache_key, self.cache_ttl_seconds, cached_data_str)
                logger.info(f"Cached predictions in Redis for {model}-{dataset} ({len(data.get('predictions', []))} items)")
            except Exception as e:
                logger.error(f"Redis cache write error: {e}")
        
        # Store in file cache as backup
        cache_file = self._get_cache_file_path(cache_key)
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cached_data, indent=2))
            logger.info(f"Cached predictions in file for {model}-{dataset} ({len(data.get('predictions', []))} items)")
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
        
        # Store in database for persistence across sessions
        try:
            await prediction_db.save_predictions(model, dataset, data, **kwargs)
            logger.info(f"Persisted predictions in database for {model}-{dataset} ({len(data.get('predictions', []))} items)")
        except Exception as e:
            logger.error(f"Database cache write error: {e}")
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid based on TTL"""
        try:
            timestamp = datetime.fromisoformat(cached_data["timestamp"])
            return datetime.now() - timestamp < self.cache_ttl
        except (KeyError, ValueError):
            return False
    
    async def invalidate(self, model: str = None, dataset: str = None) -> int:
        """
        Invalidate cache entries across all cache layers. 
        If model/dataset specified, only invalidate matching entries.
        Returns number of invalidated entries.
        """
        invalidated_count = 0
        
        # Clear Redis cache
        redis_client = await self._get_redis()
        if redis_client:
            try:
                # Get all prediction keys
                pattern = "predictions:*"
                keys = await redis_client.keys(pattern)
                
                for key in keys:
                    try:
                        cached_data_str = await redis_client.get(key)
                        if cached_data_str:
                            cached_data = json.loads(cached_data_str)
                            if self._should_invalidate(cached_data, model, dataset):
                                await redis_client.delete(key)
                                invalidated_count += 1
                    except Exception as e:
                        logger.error(f"Error processing Redis key {key}: {e}")
                        
            except Exception as e:
                logger.error(f"Redis cache invalidation error: {e}")
        
        # Clear file cache
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file, 'r') as f:
                        content = await f.read()
                        cached_data = json.loads(content)
                    
                    if self._should_invalidate(cached_data, model, dataset):
                        cache_file.unlink()
                        invalidated_count += 1
                except Exception as e:
                    logger.error(f"Error processing cache file {cache_file}: {e}")
        
        # Clear database cache
        try:
            db_deleted = await prediction_db.delete_predictions(model, dataset)
            invalidated_count += db_deleted
        except Exception as e:
            logger.error(f"Database cache invalidation error: {e}")
        
        logger.info(f"Invalidated {invalidated_count} cache entries across all cache layers")
        return invalidated_count
    
    def _should_invalidate(self, cached_data: Dict, model: str = None, dataset: str = None) -> bool:
        """Check if a cache entry should be invalidated based on model/dataset filters"""
        if model and cached_data.get("model") != model:
            return False
        if dataset and cached_data.get("dataset") != dataset:
            return False
        return True
    
    async def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics from all cache layers"""
        redis_count = 0
        redis_size = 0
        
        # Redis stats
        redis_client = await self._get_redis()
        if redis_client:
            try:
                keys = await redis_client.keys("predictions:*")
                redis_count = len(keys)
                
                # Estimate Redis memory usage
                for key in keys:
                    try:
                        size = await redis_client.memory_usage(key)
                        if size:
                            redis_size += size
                    except:
                        # Fallback to string length estimation
                        value = await redis_client.get(key)
                        if value:
                            redis_size += len(value.encode('utf-8'))
                            
            except Exception as e:
                logger.error(f"Redis stats error: {e}")
        
        # File cache stats
        file_count = len(list(self.cache_dir.glob("*.json"))) if self.cache_dir.exists() else 0
        file_size = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                file_size += cache_file.stat().st_size
        
        # Database stats
        db_stats = {}
        try:
            db_stats = await prediction_db.get_database_stats()
        except Exception as e:
            logger.error(f"Database stats error: {e}")
            db_stats = {
                'prediction_records': 0,
                'total_files': 0,
                'database_size_bytes': 0,
                'database_size_mb': 0,
                'database_path': 'N/A',
                'models': [],
                'datasets': []
            }
        
        total_size = redis_size + file_size + db_stats.get('database_size_bytes', 0)
        
        return {
            "redis_entries": redis_count,
            "redis_size_bytes": redis_size,
            "redis_size_mb": round(redis_size / 1024 / 1024, 2),
            "file_entries": file_count,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "database_entries": db_stats.get('prediction_records', 0),
            "database_files": db_stats.get('total_files', 0),
            "database_size_bytes": db_stats.get('database_size_bytes', 0),
            "database_size_mb": db_stats.get('database_size_mb', 0),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "cache_dir": str(self.cache_dir.absolute()),
            "database_path": db_stats.get('database_path', 'N/A'),
            "redis_available": redis_client is not None,
            "models": db_stats.get('models', []),
            "datasets": db_stats.get('datasets', [])
        }
    
    async def list_cached_predictions(self) -> List[Dict]:
        """List all cached predictions from database"""
        try:
            return await prediction_db.list_predictions()
        except Exception as e:
            logger.error(f"Error listing cached predictions: {e}")
            return []
    
    async def export_database(self, output_path: str = None) -> str:
        """Export database to SQL file"""
        try:
            return await prediction_db.export_to_sql(output_path)
        except Exception as e:
            logger.error(f"Error exporting database: {e}")
            raise

# Global cache instance
prediction_cache = PredictionCache()
