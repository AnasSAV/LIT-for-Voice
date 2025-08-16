#!/usr/bin/env python3
"""
RAVDESS Dataset Demo Script

Shows what the RAVDESS dataset looks like without downloading the full dataset.
This script demonstrates the structure, filenames, and metadata.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the backend app to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_ravdess import parse_ravdess_filename

def show_filename_examples():
    """Show example RAVDESS filenames and their parsed metadata."""
    print("🎵 RAVDESS Filename Examples")
    print("=" * 60)
    
    # Example filenames from RAVDESS dataset
    example_files = [
        "03-01-01-01-01-01-01.wav",  # Neutral, normal, male
        "03-01-03-01-01-01-12.wav", # Happy, normal, female
        "03-01-05-02-02-01-08.wav", # Angry, strong, female
        "03-01-06-01-01-02-15.wav", # Fearful, normal, male
        "03-01-07-02-02-01-22.wav", # Disgust, strong, female
        "03-01-08-01-01-01-04.wav", # Surprised, normal, female
    ]
    
    for filename in example_files:
        metadata = parse_ravdess_filename(filename)
        if metadata:
            print(f"\n📄 {filename}")
            print(f"   🎭 Emotion: {metadata['emotion']}")
            print(f"   👤 Actor: {metadata['actor']} ({metadata['gender']})")
            print(f"   💪 Intensity: {metadata['intensity']}")
            print(f"   💬 Statement: {metadata['statement']}")
            print(f"   🎤 Channel: {metadata['vocal_channel']}")

def show_dataset_structure():
    """Show the complete RAVDESS dataset structure."""
    print("\n🗂️ RAVDESS Dataset Structure")
    print("=" * 60)
    
    print("📊 Dataset Overview:")
    print("   • Total files: 1,440")
    print("   • Actors: 24 (12 male, 12 female)")
    print("   • Emotions: 8 categories")
    print("   • Statements: 2 different sentences")
    print("   • Repetitions: 2 per statement")
    print("   • File format: WAV audio files")
    
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    print(f"\n🎭 Emotions: {', '.join(emotions)}")
    
    print("\n📝 Statements:")
    print("   1. 'Kids are talking by the door'")
    print("   2. 'Dogs are sitting by the door'")
    
    print("\n💪 Intensities:")
    print("   • Normal (01)")
    print("   • Strong (02)")

def show_filename_format():
    """Explain the RAVDESS filename format."""
    print("\n📋 Filename Format")
    print("=" * 60)
    
    print("Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav")
    print("\nExample: 03-01-06-01-02-01-12.wav")
    print("   03: Audio-only")
    print("   01: Speech")  
    print("   06: Fearful")
    print("   01: Normal intensity")
    print("   02: 'Dogs are sitting by the door'")
    print("   01: First repetition")
    print("   12: Actor 12 (female)")
    
    print("\n🔢 Code Mappings:")
    print("Modality: 01=full-AV, 02=video-only, 03=audio-only")
    print("VocalChannel: 01=speech, 02=song")
    print("Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised")
    print("Intensity: 01=normal, 02=strong")
    print("Statement: 01='Kids are talking...', 02='Dogs are sitting...'")
    print("Repetition: 01=first, 02=second")
    print("Actor: 01-24 (odd=male, even=female)")

def create_sample_metadata_csv():
    """Create a sample CSV showing what the metadata looks like."""
    sample_data = [
        {"filename": "03-01-01-01-01-01-01.wav", "emotion": "neutral", "actor": 1, "gender": "male", "intensity": "normal", "statement": "Kids are talking by the door"},
        {"filename": "03-01-03-01-01-01-12.wav", "emotion": "happy", "actor": 12, "gender": "female", "intensity": "normal", "statement": "Kids are talking by the door"},
        {"filename": "03-01-05-02-02-01-08.wav", "emotion": "angry", "actor": 8, "gender": "female", "intensity": "strong", "statement": "Dogs are sitting by the door"},
        {"filename": "03-01-06-01-01-02-15.wav", "emotion": "fearful", "actor": 15, "gender": "male", "intensity": "normal", "statement": "Kids are talking by the door"},
        {"filename": "03-01-07-02-02-01-22.wav", "emotion": "disgust", "actor": 22, "gender": "female", "intensity": "strong", "statement": "Dogs are sitting by the door"},
        {"filename": "03-01-08-01-01-01-04.wav", "emotion": "surprised", "actor": 4, "gender": "female", "intensity": "normal", "statement": "Kids are talking by the door"},
    ]
    
    df = pd.DataFrame(sample_data)
    
    print("\n📋 Sample Metadata CSV")
    print("=" * 60)
    print(df.to_string(index=False))
    
    return df

def show_distribution():
    """Show what the emotion/gender distribution looks like."""
    print("\n📈 Dataset Distribution")
    print("=" * 60)
    
    # Each actor does: 8 emotions × 2 statements × 2 repetitions = 32 files
    # 24 actors × 32 files = 768 speech files
    # Plus 672 song files = 1,440 total
    
    print("Per Actor: 60 recordings each")
    print("   • Speech: 32 files (8 emotions × 2 statements × 2 repetitions)")
    print("   • Song: 28 files (7 emotions × 2 statements × 2 repetitions)")
    print("   • Note: No 'neutral' emotion in song files")
    
    print("\nGender Balance:")
    print("   • Male actors: 12 (odd numbers: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)")  
    print("   • Female actors: 12 (even numbers: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24)")
    
    print("\nEmotion Distribution (per modality):")
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    for emotion in emotions:
        if emotion == "neutral":
            print(f"   • {emotion}: 96 files (speech only)")
        else:
            print(f"   • {emotion}: 192 files (96 speech + 96 song)")

def main():
    print("🎵 RAVDESS Dataset Preview")
    print("=" * 80)
    print("This demo shows what the RAVDESS dataset looks like")
    print("without needing to download the full 5GB dataset first.")
    print("=" * 80)
    
    # Show the structure and format
    show_dataset_structure()
    show_filename_format()
    show_filename_examples()
    
    # Create and show sample metadata
    sample_df = create_sample_metadata_csv()
    
    # Show distribution
    show_distribution()
    
    print("\n🚀 Next Steps to Get Real Data:")
    print("=" * 60)
    print("1. Set up Kaggle API credentials:")
    print("   • Go to https://www.kaggle.com/account")
    print("   • Create API token → download kaggle.json")
    print(f"   • Place in: C:\\Users\\{os.getenv('USERNAME', 'YOUR_USER')}\\.kaggle\\kaggle.json")
    print()
    print("2. Download dataset:")
    print("   • Full dataset: python scripts/setup_ravdess.py")
    print("   • Subset only: python scripts/setup_ravdess.py --subset-only")
    print()
    print("3. Explore real data:")
    print("   • python scripts/sample_ravdess_usage.py")
    
    print("\n✨ This preview shows the exact structure of the RAVDESS dataset!")

if __name__ == "__main__":
    main()
