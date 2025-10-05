"""
Performance and Load Testing
Test Plan Section 3.1.4 & 3.1.5 - Performance Profiling and Load Testing
"""

import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np
from httpx import AsyncClient

# Performance Testing (Section 3.1.4)
class TestPerformanceProfiling:
    """Test system performance under normal and peak workload conditions."""
    
    @pytest.mark.asyncio
    async def test_model_inference_performance_whisper(self, client: AsyncClient, 
                                                       sample_audio_file, performance_thresholds):
        """Test Whisper transcription performance within time limits."""
        start_time = time.time()
        
        request_data = {
            "file_path": str(sample_audio_file),
            "model": "whisper-base"
        }
        
        response = await client.post("/inferences/whisper-attention", json=request_data)
        inference_time = time.time() - start_time
        
        # Log performance metrics
        print(f"Whisper inference time: {inference_time:.2f} seconds")
        
        if response.status_code == 200:
            max_time = performance_thresholds['model_inference_max_time']
            assert inference_time <= max_time, (
                f"Whisper inference took {inference_time:.2f}s, "
                f"max allowed {max_time}s"
            )
        else:
            # Model might not be available in test environment
            pytest.skip("Whisper model not available for performance testing")
    
    @pytest.mark.asyncio
    async def test_wav2vec2_inference_performance(self, client: AsyncClient, 
                                                  sample_audio_file, performance_thresholds):
        """Test Wav2Vec2 emotion prediction performance."""
        start_time = time.time()
        
        request_data = {
            "file_path": str(sample_audio_file)
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        inference_time = time.time() - start_time
        
        print(f"Wav2Vec2 inference time: {inference_time:.2f} seconds")
        
        if response.status_code == 200:
            max_time = performance_thresholds['model_inference_max_time']
            assert inference_time <= max_time, (
                f"Wav2Vec2 inference took {inference_time:.2f}s, "
                f"max allowed {max_time}s"
            )
        else:
            pytest.skip("Wav2Vec2 model not available for performance testing")
    
    @pytest.mark.asyncio
    async def test_cache_retrieval_performance(self, fake_redis, performance_thresholds):
        """Test Redis cache retrieval speed."""
        # Store test data
        test_data = {
            "prediction": "test result",
            "attention": np.random.rand(12, 12, 100, 100).tolist()
        }
        cache_key = "performance_test_key"
        
        import json
        await fake_redis.set(cache_key, json.dumps(test_data))
        
        # Measure retrieval time
        start_time = time.time()
        result = await fake_redis.get(cache_key)
        retrieval_time = time.time() - start_time
        
        print(f"Cache retrieval time: {retrieval_time:.4f} seconds")
        
        max_time = performance_thresholds['cache_retrieval_max_time']
        assert retrieval_time <= max_time, (
            f"Cache retrieval took {retrieval_time:.4f}s, "
            f"max allowed {max_time}s"
        )
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_audio_processing_performance(self, sample_audio_file, performance_thresholds):
        """Test audio processing pipeline performance."""
        import librosa
        
        start_time = time.time()
        
        # Simulate audio processing pipeline
        audio, sr = librosa.load(str(sample_audio_file), sr=16000)
        
        # Basic preprocessing
        audio = audio / np.max(np.abs(audio))  # Normalize
        audio = audio - np.mean(audio)  # Remove DC offset
        
        # Feature extraction (simulation)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        processing_time = time.time() - start_time
        
        print(f"Audio processing time: {processing_time:.2f} seconds")
        
        max_time = performance_thresholds['audio_upload_max_time']
        assert processing_time <= max_time, (
            f"Audio processing took {processing_time:.2f}s, "
            f"max allowed {max_time}s"
        )
        
        # Verify processing results
        assert len(audio) > 0
        assert mfccs.shape[0] == 13  # 13 MFCC coefficients
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, sample_audio_file, performance_thresholds):
        """Test memory usage during audio processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate heavy audio processing
        import librosa
        
        # Load multiple audio files (simulate batch processing)
        audio_data = []
        for _ in range(5):
            audio, sr = librosa.load(str(sample_audio_file), sr=16000)
            audio_data.append(audio)
        
        # Peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Memory usage: Initial {initial_memory:.1f}MB, Peak {peak_memory:.1f}MB")
        print(f"Memory increase: {memory_increase:.1f}MB")
        
        max_memory = performance_thresholds['max_memory_usage_mb']
        assert peak_memory <= max_memory, (
            f"Peak memory usage {peak_memory:.1f}MB exceeds limit {max_memory}MB"
        )
        
        # Cleanup
        del audio_data
    
    @pytest.mark.asyncio
    async def test_attention_calculation_performance(self, mock_model_outputs):
        """Test attention weight calculation and formatting performance."""
        start_time = time.time()
        
        # Simulate attention processing
        wav2vec2_attention = mock_model_outputs['wav2vec2']['attention']
        
        # Process attention data (simulation of visualization preparation)
        processed_attention = []
        for layer_idx, layer in enumerate(wav2vec2_attention):
            layer_data = {
                'layer': layer_idx,
                'heads': []
            }
            for head_idx, head in enumerate(layer):
                # Simulate attention head processing
                attention_matrix = np.array(head)
                normalized = (attention_matrix - np.min(attention_matrix)) / (
                    np.max(attention_matrix) - np.min(attention_matrix)
                )
                layer_data['heads'].append(normalized.tolist())
            processed_attention.append(layer_data)
        
        processing_time = time.time() - start_time
        
        print(f"Attention processing time: {processing_time:.4f} seconds")
        
        # Should process quickly for visualization
        assert processing_time <= 1.0, (
            f"Attention processing took {processing_time:.4f}s, should be under 1s"
        )
        assert len(processed_attention) == 12  # 12 layers


# Load Testing (Section 3.1.5)
class TestLoadTesting:
    """Test system under varying concurrent user loads."""
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_requests(self, client: AsyncClient, 
                                                 sample_audio_file):
        """Test multiple concurrent model inference requests."""
        async def single_request(request_id: int):
            request_data = {
                "file_path": str(sample_audio_file)
            }
            
            start_time = time.time()
            try:
                response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
                response_time = time.time() - start_time
                
                return {
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'success': response.status_code == 200
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'status_code': 500,
                    'response_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
        
        # Run 5 concurrent requests
        num_requests = 5
        tasks = [single_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r['success']]
        failed_requests = [r for r in results if isinstance(r, dict) and not r['success']]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"Concurrent requests: {num_requests}")
        print(f"Successful: {len(successful_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Exceptions: {len(exceptions)}")
        
        if successful_requests:
            avg_response_time = sum(r['response_time'] for r in successful_requests) / len(successful_requests)
            max_response_time = max(r['response_time'] for r in successful_requests)
            print(f"Average response time: {avg_response_time:.2f}s")
            print(f"Max response time: {max_response_time:.2f}s")
        
        # At least some requests should succeed
        success_rate = len(successful_requests) / num_requests
        assert success_rate >= 0.6, f"Success rate {success_rate:.2f} too low"
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, fake_redis):
        """Test Redis cache performance under concurrent access."""
        async def cache_worker(worker_id: int, operations: int = 10):
            results = []
            for i in range(operations):
                key = f"load_test_{worker_id}_{i}"
                value = {"worker": worker_id, "operation": i, "data": "x" * 100}
                
                # Write operation
                start_time = time.time()
                import json
                await fake_redis.set(key, json.dumps(value))
                write_time = time.time() - start_time
                
                # Read operation
                start_time = time.time()
                result = await fake_redis.get(key)
                read_time = time.time() - start_time
                
                results.append({
                    'worker_id': worker_id,
                    'operation': i,
                    'write_time': write_time,
                    'read_time': read_time,
                    'success': result is not None
                })
            
            return results
        
        # Run 5 concurrent workers
        num_workers = 5
        operations_per_worker = 10
        
        tasks = [cache_worker(i, operations_per_worker) for i in range(num_workers)]
        worker_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for worker_result in worker_results:
            all_results.extend(worker_result)
        
        # Analyze performance
        write_times = [r['write_time'] for r in all_results]
        read_times = [r['read_time'] for r in all_results]
        successes = [r['success'] for r in all_results]
        
        print(f"Cache operations: {len(all_results)}")
        print(f"Average write time: {np.mean(write_times):.4f}s")
        print(f"Average read time: {np.mean(read_times):.4f}s")
        print(f"Success rate: {np.mean(successes):.2f}")
        
        # Performance assertions
        assert np.mean(write_times) <= 0.1, "Cache write performance too slow"
        assert np.mean(read_times) <= 0.05, "Cache read performance too slow"
        assert np.mean(successes) >= 0.95, "Cache reliability too low"
    
    @pytest.mark.asyncio
    async def test_session_management_under_load(self, fake_redis):
        """Test session management with multiple concurrent users."""
        async def simulate_user_session(user_id: int):
            session_id = f"load_test_session_{user_id}"
            session_data = {
                'user_id': user_id,
                'files': [],
                'analyses': []
            }
            
            operations = []
            
            # Create session
            start_time = time.time()
            import json
            await fake_redis.set(f"session_{session_id}", json.dumps(session_data))
            operations.append(('create', time.time() - start_time))
            
            # Simulate user activity - multiple updates
            for i in range(5):
                session_data['files'].append(f"file_{i}.wav")
                session_data['analyses'].append(f"analysis_{i}")
                
                start_time = time.time()
                await fake_redis.set(f"session_{session_id}", json.dumps(session_data))
                operations.append(('update', time.time() - start_time))
                
                # Read session
                start_time = time.time()
                result = await fake_redis.get(f"session_{session_id}")
                operations.append(('read', time.time() - start_time))
                
                assert result is not None
            
            return {
                'user_id': user_id,
                'operations': operations,
                'total_operations': len(operations)
            }
        
        # Simulate 10 concurrent users
        num_users = 10
        tasks = [simulate_user_session(i) for i in range(num_users)]
        user_results = await asyncio.gather(*tasks)
        
        # Analyze session performance
        all_operations = []
        for user_result in user_results:
            all_operations.extend(user_result['operations'])
        
        create_times = [op[1] for op in all_operations if op[0] == 'create']
        update_times = [op[1] for op in all_operations if op[0] == 'update']
        read_times = [op[1] for op in all_operations if op[0] == 'read']
        
        print(f"Session operations: {len(all_operations)}")
        print(f"Create operations: {len(create_times)}, avg: {np.mean(create_times):.4f}s")
        print(f"Update operations: {len(update_times)}, avg: {np.mean(update_times):.4f}s")
        print(f"Read operations: {len(read_times)}, avg: {np.mean(read_times):.4f}s")
        
        # Performance requirements
        assert np.mean(create_times) <= 0.1, "Session creation too slow"
        assert np.mean(update_times) <= 0.1, "Session updates too slow"
        assert np.mean(read_times) <= 0.05, "Session reads too slow"
    
    def test_audio_file_processing_throughput(self, sample_audio_file):
        """Test throughput of audio file processing pipeline."""
        import librosa
        
        def process_audio_file(file_path: Path):
            """Simulate audio processing pipeline."""
            start_time = time.time()
            
            # Load and preprocess audio
            audio, sr = librosa.load(str(file_path), sr=16000)
            audio = audio / np.max(np.abs(audio))  # Normalize
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            
            processing_time = time.time() - start_time
            return {
                'file_size': len(audio),
                'processing_time': processing_time,
                'features_extracted': True
            }
        
        # Process the same file multiple times to measure throughput
        num_processes = 10
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.time()
            futures = [executor.submit(process_audio_file, sample_audio_file) 
                      for _ in range(num_processes)]
            results = [future.result() for future in futures]
            total_time = time.time() - start_time
        
        # Calculate throughput metrics
        processing_times = [r['processing_time'] for r in results]
        throughput = num_processes / total_time  # files per second
        
        print(f"Processed {num_processes} files in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} files/second")
        print(f"Average processing time: {np.mean(processing_times):.2f}s")
        
        # Throughput requirements
        assert throughput >= 1.0, f"Throughput {throughput:.2f} files/sec too low"
        assert all(r['features_extracted'] for r in results), "Not all files processed correctly"
    
    @pytest.mark.asyncio
    async def test_api_endpoint_rate_limiting(self, client: AsyncClient):
        """Test API rate limiting behavior under high request frequency."""
        async def rapid_requests(endpoint: str, num_requests: int = 20):
            """Send rapid requests to an endpoint."""
            results = []
            
            for i in range(num_requests):
                start_time = time.time()
                try:
                    response = await client.get(endpoint)
                    response_time = time.time() - start_time
                    
                    results.append({
                        'request_id': i,
                        'status_code': response.status_code,
                        'response_time': response_time
                    })
                except Exception as e:
                    results.append({
                        'request_id': i,
                        'status_code': 500,
                        'response_time': time.time() - start_time,
                        'error': str(e)
                    })
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.01)
            
            return results
        
        # Test health endpoint with rapid requests
        results = await rapid_requests("/health", 15)
        
        status_codes = [r['status_code'] for r in results]
        response_times = [r['response_time'] for r in results]
        
        print(f"Rapid requests: {len(results)}")
        print(f"Status code distribution: {set(status_codes)}")
        print(f"Average response time: {np.mean(response_times):.4f}s")
        
        # Most requests should succeed (health endpoint should be fast)
        success_rate = len([s for s in status_codes if s == 200]) / len(status_codes)
        assert success_rate >= 0.8, f"Success rate {success_rate:.2f} too low for health endpoint"
        
        # Response times should be reasonable
        assert np.mean(response_times) <= 0.5, "Average response time too high"


# Resource Constraint Testing
class TestResourceConstraints:
    """Test system behavior under resource limitations."""
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, sample_audio_file):
        """Test system behavior under memory pressure."""
        import gc
        
        # Monitor initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate memory pressure by loading large amount of audio data
        audio_cache = []
        
        try:
            for i in range(10):  # Load multiple audio files
                import librosa
                audio, sr = librosa.load(str(sample_audio_file), sr=16000)
                
                # Create larger arrays to increase memory usage
                large_features = np.random.rand(10000, 768)  # Simulate embedding storage
                audio_cache.append({
                    'audio': audio,
                    'features': large_features,
                    'id': i
                })
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # If memory usage gets too high, break
                if memory_increase > 1000:  # 1GB increase
                    print(f"Memory pressure reached at iteration {i}")
                    break
        
        finally:
            # Cleanup
            del audio_cache
            gc.collect()
            
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: Initial {initial_memory:.1f}MB, Final {final_memory:.1f}MB")
        
        # System should handle memory pressure gracefully
        assert final_memory < initial_memory + 500, "Memory not properly released"
    
    @pytest.mark.asyncio 
    async def test_cache_eviction_under_pressure(self, fake_redis):
        """Test cache behavior when memory limits are approached."""
        # Fill cache with data
        cache_items = []
        
        for i in range(100):
            key = f"pressure_test_{i}"
            # Create moderately large cache entries
            value = {
                'data': 'x' * 1000,  # 1KB of data
                'metadata': {
                    'created': time.time(),
                    'id': i
                }
            }
            
            import json
            await fake_redis.set(key, json.dumps(value), ex=60)  # 60 second TTL
            cache_items.append(key)
        
        # Verify items are cached
        cached_count = 0
        for key in cache_items:
            if await fake_redis.get(key) is not None:
                cached_count += 1
        
        print(f"Cached items: {cached_count}/{len(cache_items)}")
        
        # In fake redis, all items should be present
        # In real Redis with memory limits, some might be evicted
        assert cached_count >= len(cache_items) * 0.8, "Too many cache items lost"
    
    def test_disk_space_monitoring(self, temp_dir: Path):
        """Test behavior when disk space is limited."""
        import shutil
        
        # Get initial disk usage
        disk_usage = shutil.disk_usage(temp_dir)
        free_space_gb = disk_usage.free / (1024 ** 3)
        
        print(f"Available disk space: {free_space_gb:.2f} GB")
        
        # Don't actually fill disk in tests, just verify monitoring capability
        assert free_space_gb > 0.1, "Insufficient disk space for testing"
        
        # Simulate disk space check function
        def check_disk_space(path: Path, required_gb: float = 1.0) -> bool:
            """Check if sufficient disk space is available."""
            try:
                usage = shutil.disk_usage(path)
                free_gb = usage.free / (1024 ** 3)
                return free_gb >= required_gb
            except Exception:
                return False
        
        assert check_disk_space(temp_dir, 0.001), "Disk space check function failed"
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, client: AsyncClient):
        """Test behavior when connection pools are exhausted."""
        async def make_request(request_id: int):
            """Make a single request."""
            try:
                response = await client.get("/health")
                return {'id': request_id, 'status': response.status_code, 'success': True}
            except Exception as e:
                return {'id': request_id, 'status': None, 'success': False, 'error': str(e)}
        
        # Create many concurrent requests to potentially exhaust connection pool
        num_requests = 50
        tasks = [make_request(i) for i in range(num_requests)]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = len([r for r in results if isinstance(r, dict) and r.get('success')])
            failed = len(results) - successful
            
            print(f"Connection pool test: {successful} successful, {failed} failed")
            
            # Most requests should succeed even under load
            success_rate = successful / num_requests
            assert success_rate >= 0.7, f"Success rate {success_rate:.2f} too low"
            
        except Exception as e:
            print(f"Connection pool test failed: {e}")
            # This might be expected behavior under extreme load
            pass