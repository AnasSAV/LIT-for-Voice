#!/usr/bin/env python3
"""
RAVDESS Dataset Explorer

This script lets you explore the RAVDESS subset data that you've downloaded.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import librosa
import numpy as np

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_ravdess import parse_ravdess_filename

def load_subset_metadata():
    """Load the subset metadata CSV."""
    metadata_file = Path(__file__).parent.parent / "data" / "dev" / "ravdess_subset" / "ravdess_subset_metadata.csv"
    
    if not metadata_file.exists():
        print("❌ Metadata file not found. Please run the setup first.")
        return None
    
    df = pd.read_csv(metadata_file)
    print(f"📊 Loaded {len(df)} audio samples")
    print(f"   Emotions: {df['emotion'].unique().tolist()}")
    print(f"   Actors: {df['actor'].nunique()} unique")
    print(f"   Gender balance: {df['gender'].value_counts().to_dict()}")
    
    return df

def show_sample_files():
    """Show some sample filenames and their metadata."""
    subset_path = Path(__file__).parent.parent / "data" / "dev" / "ravdess_subset"
    wav_files = list(subset_path.glob("*.wav"))
    
    print(f"\n🎵 Sample Audio Files ({len(wav_files)} total):")
    print("=" * 70)
    
    for i, wav_file in enumerate(wav_files[:10], 1):
        metadata = parse_ravdess_filename(wav_file.name)
        if metadata:
            file_size = wav_file.stat().st_size / 1024  # KB
            print(f"{i:2d}. {wav_file.name}")
            print(f"    🎭 Emotion: {metadata['emotion']:>10} | 👤 Actor: {metadata['actor']:>2} ({metadata['gender']})")
            print(f"    💪 Intensity: {metadata['intensity']:>6} | 📁 Size: {file_size:.1f} KB")
            print()

def load_and_analyze_audio(filename: str):
    """Load and analyze a specific audio file."""
    audio_path = Path(__file__).parent.parent / "data" / "dev" / "ravdess_subset" / filename
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {filename}")
        return None, None
    
    # Load audio with librosa
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        print(f"\n🎵 Audio Analysis: {filename}")
        print("=" * 50)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"Total samples: {len(y)}")
        print(f"Min amplitude: {y.min():.4f}")
        print(f"Max amplitude: {y.max():.4f}")
        print(f"RMS energy: {np.sqrt(np.mean(y**2)):.4f}")
        
        # Basic features
        print(f"\n📈 Audio Features:")
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        print(f"Zero crossing rate: {np.mean(zcr):.4f}")
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        print(f"Spectral centroid: {np.mean(spectral_centroids):.2f} Hz")
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"MFCCs shape: {mfccs.shape} (13 coefficients x {mfccs.shape[1]} frames)")
        
        return y, sr
        
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None, None

def explore_emotions():
    """Show distribution of emotions in the subset."""
    df = load_subset_metadata()
    if df is None:
        return
    
    print(f"\n🎭 Emotion Distribution:")
    print("=" * 40)
    emotion_counts = df['emotion'].value_counts().sort_index()
    for emotion, count in emotion_counts.items():
        print(f"{emotion:>10}: {count:>2} files")

def explore_by_actor_gender():
    """Show distribution by actor and gender."""
    df = load_subset_metadata()
    if df is None:
        return
    
    print(f"\n👥 Actor & Gender Distribution:")
    print("=" * 40)
    
    gender_counts = df['gender'].value_counts()
    print(f"Gender balance:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} files")
    
    print(f"\nActors in subset: {sorted(df['actor'].unique())}")

def interactive_menu():
    """Interactive menu to explore the dataset."""
    while True:
        print("\n🎵 RAVDESS Dataset Explorer")
        print("=" * 50)
        print("1. Show dataset overview")
        print("2. Show sample files")
        print("3. Analyze specific audio file")
        print("4. Show emotion distribution")
        print("5. Show actor/gender distribution")
        print("6. List all files")
        print("0. Exit")
        
        choice = input("\nChoose an option (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            load_subset_metadata()
        elif choice == "2":
            show_sample_files()
        elif choice == "3":
            filename = input("Enter filename (e.g., 03-01-01-01-01-01-01.wav): ").strip()
            if filename:
                load_and_analyze_audio(filename)
        elif choice == "4":
            explore_emotions()
        elif choice == "5":
            explore_by_actor_gender()
        elif choice == "6":
            subset_path = Path(__file__).parent.parent / "data" / "dev" / "ravdess_subset"
            wav_files = list(subset_path.glob("*.wav"))
            print(f"\n📁 All {len(wav_files)} files in subset:")
            for i, wav_file in enumerate(wav_files, 1):
                print(f"{i:2d}. {wav_file.name}")
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    print("🎵 RAVDESS Dataset Explorer")
    print("=" * 60)
    print("This tool helps you explore the RAVDESS audio dataset")
    print("that you've downloaded and processed.")
    print("=" * 60)
    
    # Quick overview
    df = load_subset_metadata()
    if df is None:
        print("\n❌ No dataset found. Please run:")
        print("   python scripts/setup_ravdess.py")
        return
    
    # Show a sample
    print("\n🎯 Random Sample Analysis:")
    sample_row = df.sample(n=1).iloc[0]
    filename = sample_row['filename']
    emotion = sample_row['emotion']
    actor = sample_row['actor']
    gender = sample_row['gender']
    
    print(f"Selected: {filename}")
    print(f"Emotion: {emotion} | Actor: {actor} ({gender})")
    
    load_and_analyze_audio(filename)
    
    # Interactive menu
    print(f"\n🎮 Want to explore more?")
    response = input("Enter 'y' for interactive menu, or any other key to exit: ").strip().lower()
    
    if response == 'y':
        interactive_menu()
    else:
        print("✅ Dataset exploration completed!")

if __name__ == "__main__":
    main()
