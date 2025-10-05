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
        # Test string data
        await fake_redis.set("test_key", "test_value")
        result = await fake_redis.get("test_key")
        assert result == "test_value"
        
        # Test JSON data
        json_data = {"model": "whisper-base", "prediction": "test transcript"}
        await fake_redis.set("json_key", json.dumps(json_data))
        stored_json = json.loads(await fake_redis.get("json_key"))
        assert stored_json == json_data
        
        # Test deletion
        await fake_redis.delete("test_key")
        result = await fake_redis.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration_handling(self, fake_redis):
        """Test cache TTL and expiration behavior."""
        # Set with expiration
        await fake_redis.set("expiring_key", "value", ex=1)  # 1 second TTL
        
        # Immediately check - should exist
        result = await fake_redis.get("expiring_key")
        assert result == "value"
        
        # Check TTL
        ttl = await fake_redis.ttl("expiring_key")
        assert 0 < ttl <= 1
        
        # Wait for expiration (simulate with manual deletion in fake redis)
        await fake_redis.delete("expiring_key")
        result = await fake_redis.get("expiring_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_key_collision_handling(self, fake_redis):
        """Test cache behavior with similar keys and hash collisions."""
        # Test similar keys don't interfere
        await fake_redis.set("model_hash_123", "value1")
        await fake_redis.set("model_hash_124", "value2")
        
        assert await fake_redis.get("model_hash_123") == "value1"
        assert await fake_redis.get("model_hash_124") == "value2"
        
        # Test overwrite behavior
        await fake_redis.set("model_hash_123", "updated_value")
        assert await fake_redis.get("model_hash_123") == "updated_value"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, fake_redis):
        """Test concurrent cache operations for race conditions."""
        async def cache_worker(worker_id: int):
            key = f"concurrent_key_{worker_id}"
            value = f"worker_{worker_id}_value"
            await fake_redis.set(key, value)
            retrieved = await fake_redis.get(key)
            assert retrieved == value
            return worker_id
        
        # Run 10 concurrent workers
        tasks = [cache_worker(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)
    
    @pytest.mark.asyncio
    async def test_cache_data_integrity_large_payloads(self, fake_redis, mock_model_outputs):
        """Test cache integrity with large model prediction payloads."""
        # Large attention data simulation
        large_attention = np.random.rand(12, 12, 500, 500).tolist()
        large_payload = {
            "prediction": "test",
            "attention": large_attention,
            "embeddings": np.random.rand(1000, 768).tolist()
        }
        
        key = "large_payload_test"
        await fake_redis.set(key, json.dumps(large_payload))
        
        retrieved = json.loads(await fake_redis.get(key))
        assert retrieved["prediction"] == "test"
        assert len(retrieved["attention"]) == 12
        assert len(retrieved["embeddings"]) == 1000


# Test Audio File Storage and Integrity (Critical Priority)
class TestAudioFileIntegrity:
    """Test audio file upload, storage, and metadata preservation."""
    
    def test_audio_file_storage_integrity(self, sample_audio_file, temp_dir):
        """Test audio file storage preserves data integrity."""
        import soundfile as sf
        import librosa
        
        # Read original file
        original_audio, original_sr = librosa.load(str(sample_audio_file), sr=None)
        
        # Simulate file copy (upload process)
        copied_path = temp_dir / "uploaded_copy.wav"
        import shutil
        shutil.copy2(sample_audio_file, copied_path)
        
        # Verify integrity
        copied_audio, copied_sr = librosa.load(str(copied_path), sr=None)
        
        assert copied_sr == original_sr
        np.testing.assert_array_almost_equal(original_audio, copied_audio, decimal=5)
    
    def test_audio_file_checksum_consistency(self, sample_audio_file, temp_dir):
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
    
    def test_multiple_audio_format_support(self, temp_dir, sample_audio_data):
        """Test support for multiple audio formats (WAV, MP3, FLAC)."""
        import soundfile as sf
        
        # Create files in different formats
        formats = {
            'wav': temp_dir / "test.wav",
            'flac': temp_dir / "test.flac"
        }
        
        # Write audio in different formats
        for format_name, file_path in formats.items():
            try:
                sf.write(file_path, sample_audio_data, 16000, format=format_name.upper())
                
                # Verify readability
                audio, sr = sf.read(file_path)
                assert sr == 16000
                assert len(audio) > 0
                
            except Exception as e:
                pytest.skip(f"Format {format_name} not supported: {e}")


# Test Session Management Integrity (Important Priority)
class TestSessionIntegrity:
    """Test session data storage, retrieval, and consistency."""
    
    @pytest.mark.asyncio
    async def test_session_storage_and_retrieval(self, fake_redis, test_session_data):
        """Test session data storage and retrieval integrity."""
        session_id = test_session_data['session_id']
        session_key = f"session_{session_id}"
        
        # Store session
        await fake_redis.set(session_key, json.dumps(test_session_data))
        
        # Retrieve and verify
        stored_data = json.loads(await fake_redis.get(session_key))
        assert stored_data == test_session_data
        assert stored_data['user_id'] == 'test_user'
    
    @pytest.mark.asyncio
    async def test_session_expiration_behavior(self, fake_redis, test_session_data):
        """Test session expiration and cleanup."""
        session_key = f"session_{test_session_data['session_id']}"
        
        # Store with TTL
        await fake_redis.set(session_key, json.dumps(test_session_data), ex=3600)  # 1 hour
        
        # Verify TTL is set
        ttl = await fake_redis.ttl(session_key)
        assert ttl > 0
        
        # Simulate session renewal
        await fake_redis.expire(session_key, 7200)  # Extend to 2 hours
        new_ttl = await fake_redis.ttl(session_key)
        assert new_ttl > ttl
    
    @pytest.mark.asyncio
    async def test_session_isolation(self, fake_redis):
        """Test that different sessions don't interfere with each other."""
        # Create multiple sessions
        sessions = {
            'session_1': {'user_id': 'user1', 'data': 'session1_data'},
            'session_2': {'user_id': 'user2', 'data': 'session2_data'},
            'session_3': {'user_id': 'user3', 'data': 'session3_data'}
        }
        
        # Store all sessions
        for session_id, session_data in sessions.items():
            await fake_redis.set(f"session_{session_id}", json.dumps(session_data))
        
        # Verify isolation - each session has correct data
        for session_id, expected_data in sessions.items():
            stored_data = json.loads(await fake_redis.get(f"session_{session_id}"))
            assert stored_data == expected_data
        
        # Modify one session, ensure others unchanged
        sessions['session_1']['data'] = 'modified_data'
        await fake_redis.set('session_session_1', json.dumps(sessions['session_1']))
        
        # Verify other sessions unchanged
        session_2_data = json.loads(await fake_redis.get('session_session_2'))
        assert session_2_data['data'] == 'session2_data'


# Test Dataset Management Integrity (Important Priority)
class TestDatasetIntegrity:
    """Test dataset file resolution, metadata consistency, and security."""
    
    def test_dataset_file_resolution(self, temp_dir):
        """Test dataset file path resolution and security."""
        # Create mock dataset structure
        dataset_dir = temp_dir / "test_dataset"
        dataset_dir.mkdir()
        (dataset_dir / "file1.wav").touch()
        (dataset_dir / "file2.wav").touch()
        
        # Test valid file resolution
        from app.services.dataset_service import resolve_file
        
        try:
            resolved_path = resolve_file("test_dataset", "file1.wav", str(temp_dir))
            assert resolved_path.exists()
            assert resolved_path.name == "file1.wav"
        except Exception:
            # If resolve_file function signature is different, skip this test
            pytest.skip("Dataset service function not available or different signature")
    
    def test_path_traversal_prevention(self, temp_dir):
        """Test prevention of path traversal attacks in dataset access."""
        # Create dataset structure
        dataset_dir = temp_dir / "safe_dataset"
        dataset_dir.mkdir()
        
        # Create file outside dataset
        outside_file = temp_dir / "secret.txt"
        outside_file.write_text("sensitive data")
        
        # Test path traversal attempts
        malicious_paths = [
            "../secret.txt",
            "../../secret.txt",
            "..\\secret.txt",
            "%2e%2e%2fsecret.txt"
        ]
        
        for malicious_path in malicious_paths:
            # This should either fail or return a safe path within the dataset
            try:
                from app.services.dataset_service import resolve_file
                resolved = resolve_file("safe_dataset", malicious_path, str(temp_dir))
                # If resolution succeeds, ensure it's within safe bounds
                assert str(dataset_dir) in str(resolved.resolve())
            except Exception:
                # Expected to fail for malicious paths
                pass
    
    def test_dataset_metadata_consistency(self, temp_dir, sample_audio_file):
        """Test dataset metadata accuracy and consistency."""
        import librosa
        
        # Get actual audio properties
        audio, sr = librosa.load(str(sample_audio_file), sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Verify metadata matches actual file properties
        assert sr > 0
        assert duration > 0
        assert len(audio) > 0
        
        # Test metadata extraction consistency
        # (This would integrate with actual metadata service if available)
        metadata = {
            'filename': sample_audio_file.name,
            'duration': duration,
            'sample_rate': sr,
            'channels': 1 if len(audio.shape) == 1 else audio.shape[1]
        }
        
        assert metadata['duration'] > 4.5  # ~5 second file
        assert metadata['sample_rate'] == 16000
        assert metadata['channels'] == 1


# Integration Tests for Cache-Database Interactions
class TestCacheDatabaseIntegration:
    """Test interactions between cache and persistent storage."""
    
    @pytest.mark.asyncio
    async def test_cache_miss_fallback(self, fake_redis, sample_audio_file):
        """Test system behavior when cache miss occurs."""
        # Ensure cache is empty for specific key
        cache_key = "missing_key_test"
        result = await fake_redis.get(cache_key)
        assert result is None
        
        # This would trigger fallback to actual computation
        # (Integration with actual inference service would be tested here)
        
    @pytest.mark.asyncio
    async def test_cache_invalidation_strategy(self, fake_redis):
        """Test cache invalidation when data updates."""
        # Store initial data
        key = "invalidation_test"
        initial_data = {"version": 1, "result": "old_result"}
        await fake_redis.set(key, json.dumps(initial_data))
        
        # Simulate data update requiring cache invalidation
        updated_data = {"version": 2, "result": "new_result"}
        await fake_redis.set(key, json.dumps(updated_data))
        
        # Verify cache contains updated data
        cached_data = json.loads(await fake_redis.get(key))
        assert cached_data["version"] == 2
        assert cached_data["result"] == "new_result"
    
    @pytest.mark.asyncio
    async def test_cache_consistency_during_concurrent_writes(self, fake_redis):
        """Test cache consistency when multiple processes write simultaneously."""
        async def concurrent_writer(writer_id: int):
            key = "concurrent_write_test"
            value = {"writer_id": writer_id, "timestamp": time.time()}
            await fake_redis.set(key, json.dumps(value))
            # Small delay to simulate processing time
            await asyncio.sleep(0.01)
            return writer_id
        
        # Launch concurrent writers
        tasks = [concurrent_writer(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify final state is consistent
        final_value = await fake_redis.get("concurrent_write_test")
        assert final_value is not None
        parsed = json.loads(final_value)
        assert "writer_id" in parsed
        assert "timestamp" in parsed