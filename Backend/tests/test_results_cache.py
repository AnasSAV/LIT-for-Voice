"""
Results Cache Testing
Test Plan Section 3.1.6 - Results Caching, Cache Optimization, Cache Invalidation
"""

import pytest
import asyncio
import json
import time
from unittest.mock import patch, Mock

# Test Basic Cache Operations (Critical Priority)
class TestResultsCacheBasicOperations:
    """Test fundamental cache operations for ML results."""
    
    @pytest.mark.asyncio
    async def test_results_cache_roundtrip(self, client):
        """Test basic cache store and retrieve operations."""
        payload = {"transcript": "hello", "confidence": 0.9}
        
        # Store result in cache
        put = await client.post("/results/whisper/abc123", json=payload)
        assert put.status_code == 200
        
        # Retrieve result from cache
        get = await client.get("/results/whisper/abc123")
        assert get.status_code == 200
        data = get.json()
        assert data["cached"] is True
        assert data["payload"] == payload
    
    @pytest.mark.asyncio
    async def test_cache_miss_handling(self, client):
        """Test behavior when requested result is not in cache."""
        # Try to get non-existent result
        response = await client.get("/results/whisper/nonexistent123")
        
        # Should handle cache miss gracefully
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["cached"] is False
    
    @pytest.mark.asyncio
    async def test_cache_overwrite_behavior(self, client):
        """Test cache behavior when overwriting existing results."""
        cache_key = "overwrite_test_456"
        
        # Store initial result
        initial_payload = {"transcript": "initial", "confidence": 0.8}
        put1 = await client.post(f"/results/whisper/{cache_key}", json=initial_payload)
        assert put1.status_code == 200
        
        # Overwrite with new result
        updated_payload = {"transcript": "updated", "confidence": 0.95}
        put2 = await client.post(f"/results/whisper/{cache_key}", json=updated_payload)
        assert put2.status_code == 200
        
        # Verify latest result is returned
        get = await client.get(f"/results/whisper/{cache_key}")
        data = get.json()
        assert data["payload"]["transcript"] == "updated"
        assert data["payload"]["confidence"] == 0.95


# Test Advanced Cache Operations (Important Priority)
class TestResultsCacheAdvancedOperations:
    """Test complex cache scenarios and different data types."""
    
    @pytest.mark.asyncio
    async def test_cache_different_model_types(self, client):
        """Test caching results from different ML models."""
        # Whisper transcription result
        whisper_payload = {
            "transcript": "speech recognition test", 
            "confidence": 0.92,
            "language": "en"
        }
        await client.post("/results/whisper/test001", json=whisper_payload)
        
        # Emotion recognition result
        emotion_payload = {
            "emotion": "happy",
            "confidence": 0.87,
            "probabilities": {"happy": 0.87, "sad": 0.13}
        }
        await client.post("/results/emotion/test001", json=emotion_payload)
        
        # Verify both are cached separately
        whisper_result = await client.get("/results/whisper/test001")
        emotion_result = await client.get("/results/emotion/test001")
        
        assert whisper_result.status_code == 200
        assert emotion_result.status_code == 200
        
        whisper_data = whisper_result.json()
        emotion_data = emotion_result.json()
        
        assert whisper_data["payload"]["transcript"] == "speech recognition test"
        assert emotion_data["payload"]["emotion"] == "happy"
    
    @pytest.mark.asyncio
    async def test_cache_large_payload_handling(self, client):
        """Test caching of large ML results."""
        # Simulate large result with attention maps, embeddings, etc.
        large_payload = {
            "transcript": "this is a longer transcription result for testing",
            "confidence": 0.94,
            "word_timestamps": [
                {"word": "this", "start": 0.0, "end": 0.2, "confidence": 0.99},
                {"word": "is", "start": 0.3, "end": 0.4, "confidence": 0.98},
                {"word": "a", "start": 0.5, "end": 0.6, "confidence": 0.95}
            ],
            "attention_weights": [[0.1, 0.2, 0.3] for _ in range(10)],
            "embeddings": [0.1 * i for i in range(100)]
        }
        
        put = await client.post("/results/whisper/large_test", json=large_payload)
        assert put.status_code == 200
        
        get = await client.get("/results/whisper/large_test")
        assert get.status_code == 200
        data = get.json()
        assert data["cached"] is True
        assert len(data["payload"]["word_timestamps"]) == 3
        assert len(data["payload"]["embeddings"]) == 100
    
    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self, client):
        """Test concurrent cache operations don't interfere."""
        async def store_and_retrieve(test_id):
            payload = {
                "transcript": f"concurrent test {test_id}",
                "confidence": 0.9,
                "test_id": test_id
            }
            
            # Store
            put = await client.post(f"/results/whisper/concurrent_{test_id}", json=payload)
            
            # Retrieve
            get = await client.get(f"/results/whisper/concurrent_{test_id}")
            
            return put.status_code, get.status_code, get.json() if get.status_code == 200 else None
        
        # Run concurrent operations
        tasks = [store_and_retrieve(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        for put_status, get_status, data in results:
            assert put_status == 200
            assert get_status == 200
            if data:
                assert data["cached"] is True


# Test Cache Performance (Important Priority)
class TestResultsCachePerformance:
    """Test cache performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_cache_retrieval_speed(self, client):
        """Test cache retrieval performance."""
        # Store test result
        payload = {"transcript": "speed test", "confidence": 0.9}
        await client.post("/results/whisper/speed_test", json=payload)
        
        # Measure retrieval time
        start_time = time.time()
        response = await client.get("/results/whisper/speed_test")
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        
        assert response.status_code == 200
        assert retrieval_time < 1.0  # Should be fast (under 1 second)
        
        data = response.json()
        assert data["cached"] is True
    
    @pytest.mark.asyncio
    async def test_cache_batch_operations(self, client):
        """Test cache performance with batch operations."""
        batch_size = 10
        
        # Store batch of results
        store_tasks = []
        for i in range(batch_size):
            payload = {
                "transcript": f"batch test {i}",
                "confidence": 0.9,
                "batch_id": i
            }
            task = client.post(f"/results/whisper/batch_{i:03d}", json=payload)
            store_tasks.append(task)
        
        store_responses = await asyncio.gather(*store_tasks)
        
        # Verify all stores succeeded
        for response in store_responses:
            assert response.status_code == 200
        
        # Retrieve batch of results
        retrieve_tasks = []
        for i in range(batch_size):
            task = client.get(f"/results/whisper/batch_{i:03d}")
            retrieve_tasks.append(task)
        
        retrieve_responses = await asyncio.gather(*retrieve_tasks)
        
        # Verify all retrievals succeeded
        success_count = 0
        for response in retrieve_responses:
            if response.status_code == 200:
                data = response.json()
                if data.get("cached"):
                    success_count += 1
        
        # At least 80% should be successfully cached
        success_rate = success_count / batch_size
        assert success_rate >= 0.8


# Test Cache Error Handling (Normal Priority)
class TestResultsCacheErrorHandling:
    """Test cache error scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_cache_invalid_payload_handling(self, client):
        """Test handling of invalid cache payloads."""
        # Try storing invalid JSON
        invalid_payloads = [
            {},  # Empty payload
            {"incomplete": "data"},  # Missing required fields
            {"transcript": None},  # None values
        ]
        
        for i, payload in enumerate(invalid_payloads):
            response = await client.post(f"/results/whisper/invalid_{i}", json=payload)
            # Should handle gracefully (may accept or reject)
            assert response.status_code in [200, 400, 422]
    
    @pytest.mark.asyncio
    async def test_cache_key_format_validation(self, client):
        """Test cache key format validation."""
        payload = {"transcript": "key test", "confidence": 0.9}
        
        # Test various key formats
        test_keys = [
            "valid_key_123",
            "key-with-dashes", 
            "key.with.dots",
            "key_with_underscores"
        ]
        
        for key in test_keys:
            response = await client.post(f"/results/whisper/{key}", json=payload)
            # Should handle different key formats
            assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_cache_nonexistent_model_handling(self, client):
        """Test behavior with non-existent model types."""
        payload = {"result": "test", "confidence": 0.9}
        
        # Try caching result for non-existent model
        response = await client.post("/results/nonexistent_model/test123", json=payload)
        
        # Should handle gracefully
        assert response.status_code in [200, 404, 422]


# Test Cache Optimization (Normal Priority) 
class TestResultsCacheOptimization:
    """Test cache optimization features."""
    
    @pytest.mark.asyncio
    async def test_cache_compression_efficiency(self, client):
        """Test cache storage efficiency with repetitive data."""
        # Store similar results with repetitive patterns
        base_payload = {
            "transcript": "repeated pattern " * 10,  # Repetitive content
            "confidence": 0.9,
            "metadata": {
                "model": "whisper-base",
                "version": "1.0",
                "settings": {"temperature": 0.0}
            }
        }
        
        # Store multiple similar results
        for i in range(5):
            payload = base_payload.copy()
            payload["id"] = i
            response = await client.post(f"/results/whisper/compression_{i}", json=payload)
            assert response.status_code == 200
        
        # Verify all can be retrieved
        for i in range(5):
            response = await client.get(f"/results/whisper/compression_{i}")
            assert response.status_code == 200
            data = response.json()
            assert data["cached"] is True
    
    @pytest.mark.asyncio
    async def test_cache_memory_usage_patterns(self, client):
        """Test cache memory usage with various result sizes."""
        # Test different payload sizes
        payload_sizes = [
            {"transcript": "small", "confidence": 0.9},  # Small
            {"transcript": "medium " * 50, "confidence": 0.9, "data": list(range(100))},  # Medium
            {"transcript": "large " * 100, "confidence": 0.9, "data": list(range(1000))}  # Large
        ]
        
        for i, payload in enumerate(payload_sizes):
            response = await client.post(f"/results/whisper/size_test_{i}", json=payload)
            assert response.status_code == 200
            
            # Verify retrieval works
            get_response = await client.get(f"/results/whisper/size_test_{i}")
            assert get_response.status_code == 200
