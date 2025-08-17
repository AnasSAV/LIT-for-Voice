#!/usr/bin/env python3
"""
RAVDESS Dataset Download Script

This script downloads the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) 
dataset from Kaggle and organizes it for use in the LIT-for-Voice project.

Requirements:
- Kaggle CLI installed and configured
- Internet connection
- Approximately 5GB of free disk space

Usage:
    python scripts/download_ravdess.py [--skip-download] [--subset-only]
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path
from typing import Optional
import subprocess
import logging

# Add the backend app to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BACKEND_ROOT = Path(__file__).parent.parent
DATA_ROOT = BACKEND_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw" / "ravdess_full"
DEV_DATA_PATH = DATA_ROOT / "dev" / "ravdess_subset"
KAGGLE_DATASET = "uwrfkaggler/ravdess-emotional-speech-audio"


def check_kaggle_auth() -> bool:
    """Check if Kaggle API is properly configured."""
    try:
        result = subprocess.run(["kaggle", "competitions", "list", "-v"], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            logger.info("✅ Kaggle API is properly configured")
            return True
        else:
            logger.error("❌ Kaggle API authentication failed")
            logger.error("Please set up your Kaggle API credentials:")
            logger.error("1. Go to https://www.kaggle.com/account")
            logger.error("2. Create an API token")
            logger.error("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
            return False
    except FileNotFoundError:
        logger.error("❌ Kaggle CLI not found. Please install with: pip install kaggle")
        return False


def download_ravdess(force: bool = False) -> bool:
    """Download RAVDESS dataset from Kaggle."""
    if RAW_DATA_PATH.exists() and not force:
        logger.info(f"✅ RAVDESS dataset already exists at {RAW_DATA_PATH}")
        return True
    
    logger.info("📥 Downloading RAVDESS dataset from Kaggle...")
    logger.info(f"Dataset: {KAGGLE_DATASET}")
    logger.info(f"Destination: {RAW_DATA_PATH}")
    
    # Create raw data directory
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the dataset
        cmd = [
            "kaggle", "datasets", "download", 
            KAGGLE_DATASET,
            "-p", str(RAW_DATA_PATH),
            "--unzip"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("✅ RAVDESS dataset downloaded successfully")
        logger.info(f"Output: {result.stdout}")
        
        # List contents
        audio_files = list(RAW_DATA_PATH.glob("**/*.wav"))
        logger.info(f"📊 Found {len(audio_files)} audio files")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download RAVDESS dataset: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during download: {e}")
        return False


def parse_ravdess_filename(filename: str) -> Optional[dict]:
    """
    Parse RAVDESS filename to extract metadata.
    
    Filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Example: 03-01-06-01-02-01-12.wav
    
    Returns:
        dict: Parsed metadata or None if parsing fails
    """
    try:
        # Remove extension and split by dashes
        name_parts = Path(filename).stem.split('-')
        
        if len(name_parts) != 7:
            logger.warning(f"⚠️ Unexpected filename format: {filename}")
            return None
        
        modality_map = {
            '01': 'full-AV', '02': 'video-only', '03': 'audio-only'
        }
        
        vocal_channel_map = {
            '01': 'speech', '02': 'song'
        }
        
        emotion_map = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        intensity_map = {
            '01': 'normal', '02': 'strong'
        }
        
        statement_map = {
            '01': 'Kids are talking by the door',
            '02': 'Dogs are sitting by the door'
        }
        
        return {
            'modality': modality_map.get(name_parts[0], f'unknown_{name_parts[0]}'),
            'vocal_channel': vocal_channel_map.get(name_parts[1], f'unknown_{name_parts[1]}'),
            'emotion': emotion_map.get(name_parts[2], f'unknown_{name_parts[2]}'),
            'intensity': intensity_map.get(name_parts[3], f'unknown_{name_parts[3]}'),
            'statement': statement_map.get(name_parts[4], f'statement_{name_parts[4]}'),
            'repetition': int(name_parts[5]),
            'actor': int(name_parts[6]),
            'gender': 'female' if int(name_parts[6]) % 2 == 0 else 'male'
        }
        
    except Exception as e:
        logger.error(f"❌ Error parsing filename {filename}: {e}")
        return None


def verify_dataset() -> bool:
    """Verify the downloaded dataset integrity."""
    if not RAW_DATA_PATH.exists():
        logger.error("❌ Raw dataset path does not exist")
        return False
    
    audio_files = list(RAW_DATA_PATH.glob("**/*.wav"))
    logger.info(f"📊 Verification: Found {len(audio_files)} audio files")
    
    # Expected: 1440 files (24 actors × 60 recordings each)
    expected_count = 1440
    
    if len(audio_files) < expected_count * 0.9:  # Allow 10% tolerance
        logger.warning(f"⚠️ Expected ~{expected_count} files, found {len(audio_files)}")
    
    # Test parsing a few filenames
    parsed_count = 0
    for audio_file in audio_files[:10]:
        metadata = parse_ravdess_filename(audio_file.name)
        if metadata:
            parsed_count += 1
    
    if parsed_count < 5:
        logger.error("❌ Failed to parse most filenames - check filename format")
        return False
    
    logger.info("✅ Dataset verification completed successfully")
    return True


def create_gitignore():
    """Create .gitignore for data directories."""
    gitignore_path = DATA_ROOT / ".gitignore"
    
    gitignore_content = """# Ignore large dataset files but keep directory structure
raw/
processed/
*.wav
*.mp3
*.flac
*.m4a

# Keep subset for development (add if needed)
!dev/ravdess_subset/.gitkeep
!dev/sample_files/

# Keep metadata and logs
!*.csv
!*.json
!*.log
"""
    
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    logger.info(f"📝 Created .gitignore at {gitignore_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and setup RAVDESS dataset")
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip download and only verify existing dataset")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if dataset exists")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify the existing dataset")
    
    args = parser.parse_args()
    
    logger.info("🎵 RAVDESS Dataset Setup")
    logger.info("=" * 50)
    
    # Check Kaggle authentication
    if not args.verify_only and not check_kaggle_auth():
        sys.exit(1)
    
    # Download dataset
    if not args.skip_download and not args.verify_only:
        if not download_ravdess(force=args.force):
            sys.exit(1)
    
    # Verify dataset
    if not verify_dataset():
        logger.error("❌ Dataset verification failed")
        sys.exit(1)
    
    # Create .gitignore
    create_gitignore()
    
    logger.info("✅ RAVDESS dataset setup completed successfully!")
    logger.info(f"📁 Dataset location: {RAW_DATA_PATH}")
    logger.info("🚀 Next steps:")
    logger.info("   1. Run: python scripts/create_ravdess_subset.py")
    logger.info("   2. Set up database and preprocessing pipeline")


if __name__ == "__main__":
    main()
