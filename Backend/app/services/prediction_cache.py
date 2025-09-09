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
        
        # Fallback to file cache
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
        
        logger.info(f"Cache miss for {model}-{dataset}")
        return None
    
    async def set(self, model: str, dataset: str, data: Dict, **kwargs) -> None:
        """
        Store predictions in cache for a model-dataset combination
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
        
        # Store in Redis first
        redis_client = await self._get_redis()
        if redis_client:
            try:
                await redis_client.setex(cache_key, self.cache_ttl_seconds, cached_data_str)
                logger.info(f"Cached predictions in Redis for {model}-{dataset} ({len(data.get('predictions', []))} items)")
            except Exception as e:
                logger.error(f"Redis cache write error: {e}")
        
        # Also store in file cache as backup
        cache_file = self._get_cache_file_path(cache_key)
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cached_data, indent=2))
            logger.info(f"Cached predictions in file for {model}-{dataset} ({len(data.get('predictions', []))} items)")
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
    
    def _is_cache_valid(self, cached_data: Dict) -> bool:
        """Check if cached data is still valid based on TTL"""
        try:
            timestamp = datetime.fromisoformat(cached_data["timestamp"])
            return datetime.now() - timestamp < self.cache_ttl
        except (KeyError, ValueError):
            return False
    
    async def invalidate(self, model: str = None, dataset: str = None) -> int:
        """
        Invalidate cache entries. If model/dataset specified, only invalidate matching entries.
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
        
        logger.info(f"Invalidated {invalidated_count} cache entries")
        return invalidated_count
    
    def _should_invalidate(self, cached_data: Dict, model: str = None, dataset: str = None) -> bool:
        """Check if a cache entry should be invalidated based on model/dataset filters"""
        if model and cached_data.get("model") != model:
            return False
        if dataset and cached_data.get("dataset") != dataset:
            return False
        return True
    
    async def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
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
        
        return {
            "redis_entries": redis_count,
            "redis_size_bytes": redis_size,
            "redis_size_mb": round(redis_size / 1024 / 1024, 2),
            "file_entries": file_count,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "total_size_mb": round((redis_size + file_size) / 1024 / 1024, 2),
            "cache_dir": str(self.cache_dir.absolute()),
            "redis_available": redis_client is not None
        }

# Global cache instance
prediction_cache = PredictionCache()
