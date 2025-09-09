import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.model_loader_service import predict_emotion_wave2vec, transcribe_whisper_base
from pathlib import Path

async def test_predictions():
    """Test script to verify predictions work"""
    
    # Get a sample file from ravdess dataset
    ravdess_dir = Path("data/ravdess_subset")
    
    # List first few files
    files = list(ravdess_dir.glob("*.wav"))[:3]
    
    if not files:
        print("No audio files found in ravdess_subset directory")
        return
    
    print(f"Testing predictions on {len(files)} files...")
    
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        
        try:
            # Test emotion prediction
            emotion = predict_emotion_wave2vec(str(file_path))
            print(f"  Predicted emotion: {emotion}")
            
            # Test transcript extraction
            transcript = transcribe_whisper_base(str(file_path))
            print(f"  Transcript: {transcript}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_predictions())
