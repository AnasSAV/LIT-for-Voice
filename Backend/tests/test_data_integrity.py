"""
Data and Database Integrity Testing
Test Plan Section 3.1.1 - Redis Cache, Audio File Storage, Session Management
"""

import pytest
import asyncio
import json
import hashlib
import time
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np

# Test Redis Cache Operations (Critical Priority)
class TestRedisCacheIntegrity:
    """Test Redis caching mechanisms for data integrity and consistency."""
    
    @pytest.mark.asyncio
    async def test_cache_basic_operations(self, fake_redis):
        """Test basic Redis SET, GET, DEL operations with various data types."""
        # The fake_redis fixture from conftest.py doesn't return the client directly
        # We need to access the redis module that was patched
        from app.core import redis as redis_module
        redis_client = redis_module.redis
        
        # Test string data
        await redis_client.set("test_key", "test_value")
        result = await redis_client.get("test_key")
        assert result == "test_value"
        
        # Test JSON data
        json_data = {"model": "whisper-base", "prediction": "test transcript"}
        await redis_client.set("json_key", json.dumps(json_data))
        stored_json = json.loads(await redis_client.get("json_key"))
        assert stored_json == json_data
        
        # Test deletion
        await redis_client.delete("test_key")
        result = await redis_client.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration_handling(self, fake_redis):
        """Test cache TTL and expiration behavior."""
        from app.core import redis as redis_module
        redis_client = redis_module.redis
        
        # Set with expiration
        await redis_client.set("expiring_key", "value", ex=1)  # 1 second TTL
        
        # Immediately check - should exist
        result = await redis_client.get("expiring_key")
        assert result == "value"
        
        # Check TTL
        ttl = await redis_client.ttl("expiring_key")
        assert 0 < ttl <= 1
    
    @pytest.mark.asyncio
    async def test_cache_key_collision_handling(self, fake_redis):
        """Test cache behavior with similar keys and hash collisions."""
        from app.core import redis as redis_module
        redis_client = redis_module.redis
        
        # Test similar keys don't interfere
        await redis_client.set("model_hash_123", "value1")
        await redis_client.set("model_hash_124", "value2")
        
        assert await redis_client.get("model_hash_123") == "value1"
        assert await redis_client.get("model_hash_124") == "value2"
        
        # Test overwrite behavior
        await redis_client.set("model_hash_123", "updated_value")
        assert await redis_client.get("model_hash_123") == "updated_value"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, fake_redis):
        """Test concurrent cache operations for race conditions."""
        from app.core import redis as redis_module
        redis_client = redis_module.redis
        
        async def cache_worker(worker_id: int):
            key = f"concurrent_key_{worker_id}"
            value = f"worker_{worker_id}_value"
            await redis_client.set(key, value)
            retrieved = await redis_client.get(key)
            assert retrieved == value
            return worker_id
        
        # Run 10 concurrent workers
        tasks = [cache_worker(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)


# Test Audio File Storage and Integrity (Critical Priority)
class TestAudioFileIntegrity:
    """Test audio file upload, storage, and metadata preservation."""
    
    def test_audio_file_checksum_consistency(self, temp_dir, sample_audio_file):
        """Test file checksum consistency after storage operations."""
        # Calculate original checksum
        with open(sample_audio_file, 'rb') as f:
            original_hash = hashlib.md5(f.read()).hexdigest()
        
        # Copy file (simulate upload)
        copied_path = temp_dir / "checksum_test.wav"
        import shutil
        shutil.copy2(sample_audio_file, copied_path)
        
        # Calculate copied checksum
        with open(copied_path, 'rb') as f:
            copied_hash = hashlib.md5(f.read()).hexdigest()
        
        assert original_hash == copied_hash
    
    def test_audio_metadata_preservation(self, sample_audio_file):
        """Test audio metadata preservation during processing."""
        import librosa
        
        # Get metadata
        audio, sr = librosa.load(str(sample_audio_file), sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Verify expected properties
        assert sr == 16000  # Expected sample rate
        assert 4.5 < duration < 5.5  # ~5 seconds audio
        assert len(audio.shape) == 1  # Mono audio
        assert audio.dtype == np.float32


# Test Session Management Integrity (Important Priority)
class TestSessionIntegrity:
    """Test session data storage, retrieval, and consistency."""
    
    @pytest.mark.asyncio
    async def test_session_storage_and_retrieval(self, fake_redis, test_session_data):
        """Test session data storage and retrieval integrity."""
        from app.core import redis as redis_module
        redis_client = redis_module.redis
        
        session_id = test_session_data['session_id']
        session_key = f"session_{session_id}"
        
        # Store session
        await redis_client.set(session_key, json.dumps(test_session_data))
        
        # Retrieve and verify
        stored_data = json.loads(await redis_client.get(session_key))
        assert stored_data == test_session_data
        assert stored_data['user_id'] == 'test_user'
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, fake_redis):
        """Test that different sessions don't interfere with each other."""
        from app.core import redis as redis_module
        redis_client = redis_module.redis
        
        # Create multiple sessions
        sessions = {
            'session_1': {'user_id': 'user1', 'data': 'session1_data'},
            'session_2': {'user_id': 'user2', 'data': 'session2_data'},
            'session_3': {'user_id': 'user3', 'data': 'session3_data'}
        }
        
        # Store all sessions
        for session_id, session_data in sessions.items():
            await redis_client.set(f"session_{session_id}", json.dumps(session_data))
        
        # Verify isolation - each session has correct data
        for session_id, expected_data in sessions.items():
            stored_data = json.loads(await redis_client.get(f"session_{session_id}"))
            assert stored_data == expected_data
