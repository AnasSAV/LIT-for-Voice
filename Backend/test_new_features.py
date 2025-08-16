#!/usr/bin/env python3
"""
Test script for the new optimized features:
1. Model caching
2. Redis result caching  
3. File upload support
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/inferences/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_list_models():
    """Test listing available models"""
    print("=== Testing List Models ===")
    response = requests.get(f"{BASE_URL}/inferences/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_sample_inference_with_timing():
    """Test inference on sample files with timing"""
    print("=== Testing Sample Inference (with caching) ===")
    
    model = "whisper-base"
    
    # First call - should be slow (no cache)
    print("First call (no cache):")
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/inferences/run?model={model}")
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    if response.status_code == 200:
        result = response.json()
        print(f"Result keys: {list(result.keys())}")
    print()
    
    # Second call - should be fast (cached)
    print("Second call (cached):")
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/inferences/run?model={model}")
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    if response.status_code == 200:
        result = response.json()
        print(f"Result keys: {list(result.keys())}")
    print()

def test_file_upload():
    """Test file upload functionality"""
    print("=== Testing File Upload ===")
    
    # Create a simple test audio file path (using existing sample)
    audio_file_path = "sample1.wav"
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            data = {'model': 'whisper-base'}
            
            print("Uploading file...")
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/inferences/upload", files=files, data=data)
            end_time = time.time()
            
            print(f"Status: {response.status_code}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Result keys: {list(result.keys())}")
            else:
                print(f"Error: {response.text}")
    except FileNotFoundError:
        print(f"Audio file {audio_file_path} not found. Skipping file upload test.")
    print()

def test_clear_cache():
    """Test cache clearing"""
    print("=== Testing Cache Clear ===")
    
    model = "whisper-base"
    response = requests.delete(f"{BASE_URL}/inferences/cache/{model}")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("Testing new optimized features...\n")
    
    test_health_check()
    test_list_models()
    test_sample_inference_with_timing()
    test_file_upload()
    test_clear_cache()
    
    print("=== Testing Complete ===")
