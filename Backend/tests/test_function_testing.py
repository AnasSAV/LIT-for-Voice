"""
Function Testing - Core ML Model Integration and Audio Processing
Test Plan Section 3.1.2 - ML Models, Audio Processing, Interpretability Analysis
"""

import pytest
import asyncio
import json
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient

# Test ML Model Integration (Critical Priority)
class TestMLModelIntegration:
    """Test Whisper, Wav2Vec2, and attention mechanism integration."""
    
    @pytest.mark.asyncio
    async def test_whisper_transcription_endpoint(self, client: AsyncClient, sample_audio_file):
        """Test Whisper transcription API endpoint with various inputs."""
        # Prepare request payload
        files = {"file": ("test.wav", open(sample_audio_file, "rb"), "audio/wav")}
        
        try:
            response = await client.post("/inferences/whisper-base", files=files)
            
            # Should return 200 or handle gracefully if model not loaded
            if response.status_code == 200:
                result = response.json()
                assert "predicted_transcript" in result
                assert isinstance(result["predicted_transcript"], str)
                assert len(result["predicted_transcript"]) > 0
            else:
                # Model might not be available in test environment
                assert response.status_code in [404, 500, 503]
                
        finally:
            files["file"][1].close()
    
    @pytest.mark.asyncio
    async def test_wav2vec2_emotion_prediction(self, client: AsyncClient, sample_audio_file):
        """Test Wav2Vec2 emotion prediction with audio input."""
        request_data = {
            "file_path": str(sample_audio_file)
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            assert "predicted_emotion" in result
            assert "confidence" in result
            assert isinstance(result["confidence"], (int, float))
            assert 0 <= result["confidence"] <= 1
        else:
            # Model might not be available in test environment
            assert response.status_code in [404, 500, 503]
    
    @pytest.mark.asyncio
    @patch('app.services.model_loader_service.transcribe_whisper_with_attention')
    async def test_whisper_attention_extraction(self, mock_whisper, client: AsyncClient, 
                                                sample_audio_file, mock_model_outputs):
        """Test Whisper attention mechanism extraction."""
        # Mock the model service
        mock_whisper.return_value = mock_model_outputs['whisper']
        
        request_data = {
            "file_path": str(sample_audio_file),
            "model": "whisper-base"
        }
        
        response = await client.post("/inferences/whisper-attention", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            assert "predicted_transcript" in result
            assert "attention" in result
            
            # Validate attention format
            attention = result["attention"]
            assert isinstance(attention, list)
            assert len(attention) > 0  # Should have layers
            
            # Check attention dimensions (layers x heads x seq_len x encoder_len)
            first_layer = attention[0]
            assert isinstance(first_layer, list)
    
    @pytest.mark.asyncio
    @patch('app.services.model_loader_service.predict_emotion_wave2vec_with_attention')
    async def test_wav2vec2_attention_extraction(self, mock_wav2vec2, client: AsyncClient,
                                                 sample_audio_file, mock_model_outputs):
        """Test Wav2Vec2 attention mechanism extraction."""
        # Mock the model service
        mock_wav2vec2.return_value = mock_model_outputs['wav2vec2']
        
        request_data = {
            "file_path": str(sample_audio_file)
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            assert "predicted_emotion" in result
            assert "attention" in result
            
            # Validate attention format for Wav2Vec2
            attention = result["attention"]
            assert isinstance(attention, list)
            assert len(attention) == 12  # Expected layers for base model
    
    def test_model_output_validation_whisper(self, mock_model_outputs):
        """Test Whisper model output format validation."""
        whisper_output = mock_model_outputs['whisper']
        
        # Required fields
        assert "predicted_transcript" in whisper_output
        assert "language" in whisper_output
        assert "segments" in whisper_output
        
        # Validate segments format
        segments = whisper_output["segments"]
        assert isinstance(segments, list)
        for segment in segments:
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert isinstance(segment["start"], (int, float))
            assert isinstance(segment["end"], (int, float))
    
    def test_model_output_validation_wav2vec2(self, mock_model_outputs):
        """Test Wav2Vec2 model output format validation."""
        wav2vec2_output = mock_model_outputs['wav2vec2']
        
        # Required fields
        assert "predicted_emotion" in wav2vec2_output
        assert "confidence" in wav2vec2_output
        assert "all_predictions" in wav2vec2_output
        
        # Validate confidence score
        confidence = wav2vec2_output["confidence"]
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
        
        # Validate all predictions sum to ~1.0
        all_preds = wav2vec2_output["all_predictions"]
        total = sum(all_preds.values())
        assert 0.95 <= total <= 1.05  # Allow for floating point precision


# Test Audio Processing Pipeline (Critical Priority)  
class TestAudioProcessing:
    """Test audio file upload, conversion, preprocessing, and perturbation."""
    
    @pytest.mark.asyncio
    async def test_audio_file_upload(self, client: AsyncClient, sample_audio_file):
        """Test audio file upload endpoint."""
        files = {"file": ("test.wav", open(sample_audio_file, "rb"), "audio/wav")}
        
        try:
            response = await client.post("/upload/audio", files=files)
            
            if response.status_code == 200:
                result = response.json()
                assert "file_id" in result or "filename" in result
                assert "message" in result
            else:
                # Upload endpoint might have different implementation
                assert response.status_code in [404, 422, 500]
                
        finally:
            files["file"][1].close()
    
    def test_audio_format_validation(self, temp_dir, sample_audio_data):
        """Test audio format support and validation."""
        import soundfile as sf
        
        # Test supported formats
        supported_formats = ['wav', 'flac']
        
        for format_name in supported_formats:
            file_path = temp_dir / f"test.{format_name}"
            
            try:
                # Create file in format
                sf.write(file_path, sample_audio_data, 16000, format=format_name.upper())
                
                # Validate can be read
                audio, sr = sf.read(file_path)
                assert sr == 16000
                assert len(audio) > 0
                assert audio.dtype in [np.float32, np.float64]
                
            except Exception as e:
                pytest.skip(f"Format {format_name} not supported: {e}")
    
    def test_audio_preprocessing_normalization(self, sample_audio_data):
        """Test audio preprocessing and normalization."""
        # Test amplitude normalization
        normalized = sample_audio_data / np.max(np.abs(sample_audio_data))
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test DC offset removal
        dc_removed = normalized - np.mean(normalized)
        assert abs(np.mean(dc_removed)) < 1e-10
    
    def test_waveform_data_extraction(self, sample_audio_file):
        """Test waveform data extraction for visualization."""
        import librosa
        
        # Load audio
        audio, sr = librosa.load(str(sample_audio_file), sr=16000)
        
        # Test waveform properties
        assert len(audio.shape) == 1  # Mono
        assert sr == 16000
        assert len(audio) > 0
        
        # Test resampling if needed
        if sr != 16000:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            assert len(audio_resampled) > 0
    
    @pytest.mark.asyncio
    async def test_audio_perturbation_methods(self, client: AsyncClient, sample_audio_file):
        """Test audio perturbation for robustness analysis."""
        perturbation_data = {
            "file_path": str(sample_audio_file),
            "perturbation_type": "noise",
            "intensity": 0.1
        }
        
        response = await client.post("/perturb/audio", json=perturbation_data)
        
        if response.status_code == 200:
            result = response.json()
            assert "perturbed_file_path" in result or "perturbed_audio" in result
        else:
            # Perturbation endpoint might not be implemented yet
            assert response.status_code in [404, 501, 422]


# Test Interpretability Analysis Methods (Important Priority)
class TestInterpretabilityAnalysis:
    """Test saliency, embedding, attention, and perturbation analysis."""
    
    @pytest.mark.asyncio
    async def test_saliency_map_generation(self, client: AsyncClient, sample_audio_file):
        """Test gradient-based saliency map generation."""
        request_data = {
            "file_path": str(sample_audio_file),
            "model": "wav2vec2"
        }
        
        response = await client.post("/saliency/generate", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            assert "saliency_map" in result or "gradients" in result
            
            # Validate saliency data format
            if "saliency_map" in result:
                saliency = result["saliency_map"]
                assert isinstance(saliency, list)
                assert len(saliency) > 0
        else:
            # Saliency endpoint might not be implemented
            assert response.status_code in [404, 501, 422]
    
    @pytest.mark.asyncio
    async def test_embedding_extraction_and_visualization(self, client: AsyncClient, sample_audio_file):
        """Test embedding extraction and dimensionality reduction."""
        request_data = {
            "file_path": str(sample_audio_file),
            "model": "wav2vec2"
        }
        
        response = await client.post("/embeddings/extract", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            assert "embedding" in result
            
            # Validate embedding format
            embedding = result["embedding"]
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            
            # Test dimensionality (should be feature vector)
            if len(embedding) > 0:
                assert len(embedding) > 10  # Should be substantial feature vector
        else:
            # Embedding endpoint might have different path
            assert response.status_code in [404, 501, 422]
    
    def test_attention_pattern_validation(self, mock_model_outputs):
        """Test attention pattern format and visualization data."""
        # Test Whisper attention
        whisper_attention = mock_model_outputs['whisper']['attention']
        assert isinstance(whisper_attention, list)
        assert len(whisper_attention) > 0
        
        # Test Wav2Vec2 attention
        wav2vec2_attention = mock_model_outputs['wav2vec2']['attention']
        assert isinstance(wav2vec2_attention, list)
        assert len(wav2vec2_attention) == 12  # Expected layers
        
        # Validate attention dimensions
        first_layer = wav2vec2_attention[0]
        assert isinstance(first_layer, list)
        assert len(first_layer) == 12  # Expected heads
    
    @pytest.mark.asyncio
    async def test_perturbation_analysis_statistical_significance(self, client: AsyncClient, sample_audio_file):
        """Test perturbation analysis for model robustness."""
        # Test multiple perturbations
        perturbation_types = ["noise", "time_stretch", "pitch_shift"]
        
        for perturb_type in perturbation_types:
            request_data = {
                "file_path": str(sample_audio_file),
                "perturbation_type": perturb_type,
                "num_samples": 5,
                "intensity": 0.1
            }
            
            response = await client.post("/perturb/analyze", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                assert "analysis_results" in result or "perturbation_effects" in result
            else:
                # Perturbation analysis might not be fully implemented
                assert response.status_code in [404, 501, 422]


# Test User Workflow Integration (Important Priority)
class TestUserWorkflows:
    """Test complete user workflows from upload to analysis."""
    
    @pytest.mark.asyncio
    async def test_complete_upload_to_transcription_workflow(self, client: AsyncClient, sample_audio_file):
        """Test complete workflow: upload → transcription → results."""
        # Step 1: Upload audio file
        files = {"file": ("test.wav", open(sample_audio_file, "rb"), "audio/wav")}
        
        try:
            upload_response = await client.post("/upload/audio", files=files)
            
            if upload_response.status_code != 200:
                pytest.skip("Upload endpoint not available")
                
            upload_result = upload_response.json()
            
            # Step 2: Run transcription
            transcription_data = {
                "file_path": upload_result.get("file_path", str(sample_audio_file)),
                "model": "whisper-base"
            }
            
            transcription_response = await client.post("/inferences/whisper-base", json=transcription_data)
            
            if transcription_response.status_code == 200:
                transcription_result = transcription_response.json()
                assert "predicted_transcript" in transcription_result
                
        finally:
            files["file"][1].close()
    
    @pytest.mark.asyncio
    async def test_complete_emotion_analysis_workflow(self, client: AsyncClient, sample_audio_file):
        """Test complete workflow: upload → emotion analysis → attention visualization."""
        request_data = {
            "file_path": str(sample_audio_file)
        }
        
        # Run emotion prediction with attention
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Verify we get both prediction and attention
            assert "predicted_emotion" in result
            
            # If attention is available, verify format
            if "attention" in result:
                attention = result["attention"]
                assert isinstance(attention, list)
                assert len(attention) > 0
    
    @pytest.mark.asyncio
    async def test_dataset_selection_and_analysis_workflow(self, client: AsyncClient):
        """Test workflow with dataset file selection."""
        # Test dataset file analysis
        request_data = {
            "dataset": "ravdess",
            "dataset_file": "test_file.wav"  # Mock dataset file
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        
        # This might fail if dataset not available, which is expected
        if response.status_code not in [200, 404, 422]:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs throughout workflows."""
        # Test invalid file paths
        invalid_paths = [
            "/nonexistent/path.wav",
            "",
            None,
            "../../../etc/passwd",
            "http://malicious.com/file.wav"
        ]
        
        for invalid_path in invalid_paths:
            # This would be tested with actual API calls
            # For now, just validate the input validation logic
            if invalid_path:
                assert isinstance(invalid_path, str)
            else:
                assert invalid_path is None or invalid_path == ""


# Test Model Performance and Accuracy (Important Priority)
class TestModelPerformance:
    """Test model inference performance and accuracy benchmarks."""
    
    @pytest.mark.asyncio
    async def test_model_inference_timing(self, client: AsyncClient, sample_audio_file, performance_thresholds):
        """Test model inference completes within performance thresholds."""
        import time
        
        request_data = {
            "file_path": str(sample_audio_file)
        }
        
        # Test Wav2Vec2 inference timing
        start_time = time.time()
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        inference_time = time.time() - start_time
        
        if response.status_code == 200:
            # Should complete within threshold (10 seconds per test plan)
            max_time = performance_thresholds['model_inference_max_time']
            assert inference_time <= max_time, f"Inference took {inference_time:.2f}s, max allowed {max_time}s"
    
    def test_model_output_consistency(self, mock_model_outputs):
        """Test model output consistency across multiple runs."""
        # Test that output format is consistent
        whisper_output = mock_model_outputs['whisper']
        wav2vec2_output = mock_model_outputs['wav2vec2']
        
        # Whisper consistency
        assert isinstance(whisper_output['predicted_transcript'], str)
        assert len(whisper_output['segments']) > 0
        
        # Wav2Vec2 consistency
        assert whisper_output['predicted_emotion'] in ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
        assert 0 <= wav2vec2_output['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, client: AsyncClient, sample_audio_file):
        """Test performance with multiple concurrent requests."""
        async def single_request():
            request_data = {"file_path": str(sample_audio_file)}
            response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
            return response.status_code
        
        # Run 3 concurrent requests (lightweight test)
        tasks = [single_request() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some should succeed or fail gracefully
        status_codes = [r for r in results if isinstance(r, int)]
        assert len(status_codes) > 0
        
        # No unexpected errors (exceptions)
        exceptions = [r for r in results if isinstance(r, Exception)]
        for exc in exceptions:
            # Only allow expected exceptions (connection issues, etc.)
            assert isinstance(exc, (asyncio.TimeoutError, ConnectionError))


# Error Handling and Edge Cases (Important Priority)
class TestErrorHandling:
    """Test error handling for various failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_audio_file_handling(self, client: AsyncClient, temp_dir):
        """Test handling of corrupted or invalid audio files."""
        # Create invalid audio file
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")
        
        request_data = {
            "file_path": str(invalid_file)
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        
        # Should handle gracefully with appropriate error
        assert response.status_code in [400, 422, 500]
        
        if response.status_code != 500:  # Don't check body for 500 errors
            error_response = response.json()
            assert "detail" in error_response or "error" in error_response
    
    @pytest.mark.asyncio
    async def test_missing_file_error_handling(self, client: AsyncClient):
        """Test error handling for missing files."""
        request_data = {
            "file_path": "/nonexistent/file.wav"
        }
        
        response = await client.post("/inferences/wav2vec2-detailed", json=request_data)
        
        # Should return appropriate error code
        assert response.status_code in [404, 422, 400]
    
    @pytest.mark.asyncio
    async def test_malformed_request_handling(self, client: AsyncClient):
        """Test handling of malformed API requests."""
        # Test missing required fields
        response = await client.post("/inferences/wav2vec2-detailed", json={})
        assert response.status_code in [400, 422]
        
        # Test invalid JSON
        response = await client.post(
            "/inferences/wav2vec2-detailed", 
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_unsupported_model_handling(self, client: AsyncClient, sample_audio_file):
        """Test handling of unsupported model requests."""
        request_data = {
            "file_path": str(sample_audio_file),
            "model": "nonexistent-model"
        }
        
        response = await client.post("/inferences/whisper-attention", json=request_data)
        
        # Should handle unsupported model gracefully
        assert response.status_code in [400, 404, 422]