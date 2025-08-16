#!/usr/bin/env python3
"""
RAVDESS Setup Script

One-stop script to set up the RAVDESS dataset integration.
This script handles:
1. Environment setup and dependencies
2. Kaggle credentials verification  
3. Dataset download
4. Subset creation
5. Initial verification

Usage:
    python scripts/setup_ravdess.py [--subset-only] [--subset-size 0.05]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BACKEND_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = BACKEND_ROOT / "scripts"


def check_dependencies():
    """Check if required Python packages are installed."""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = ['kaggle', 'librosa', 'numpy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {package}")
        except ImportError:
            logger.warning(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.info("📦 Installing missing packages...")
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
        logger.info("✅ Dependencies installed")
    else:
        logger.info("✅ All dependencies satisfied")


def setup_kaggle_instructions():
    """Provide instructions for setting up Kaggle credentials."""
    logger.info("🔑 Kaggle API Setup Instructions:")
    logger.info("=" * 50)
    logger.info("1. Go to https://www.kaggle.com/account")
    logger.info("2. Scroll to 'API' section")
    logger.info("3. Click 'Create New API Token'")
    logger.info("4. This downloads kaggle.json")
    logger.info("5. Place kaggle.json in one of these locations:")
    username = os.getenv('USERNAME', 'YOUR_USER')
    logger.info(f"   Windows: C:\\Users\\{username}\\.kaggle\\kaggle.json")
    logger.info("   Linux/Mac: ~/.kaggle/kaggle.json")
    logger.info("6. Set file permissions (Unix only): chmod 600 ~/.kaggle/kaggle.json")
    logger.info("=" * 50)


def run_download_script(force: bool = False):
    """Run the RAVDESS download script."""
    logger.info("📥 Starting RAVDESS download...")
    
    download_script = SCRIPTS_DIR / "download_ravdess.py"
    cmd = [sys.executable, str(download_script)]
    
    if force:
        cmd.append("--force")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ Download completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Download failed: {e}")
        if "401" in str(e) or "authentication" in str(e.stderr).lower():
            logger.error("🔑 This looks like a Kaggle authentication issue")
            setup_kaggle_instructions()
        return False


def run_subset_script(subset_size: float = 0.05, force: bool = False):
    """Run the RAVDESS subset creation script."""
    logger.info("🎭 Creating development subset...")
    
    subset_script = SCRIPTS_DIR / "create_ravdess_subset.py"
    cmd = [sys.executable, str(subset_script), "--size", str(subset_size)]
    
    if force:
        cmd.append("--force")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ Subset created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Subset creation failed: {e}")
        return False


def verify_setup():
    """Verify that everything is set up correctly."""
    logger.info("🔍 Verifying setup...")
    
    # Check if subset exists and has files
    subset_path = BACKEND_ROOT / "data" / "dev" / "ravdess_subset"
    if not subset_path.exists():
        logger.error("❌ Subset directory not found")
        return False
    
    wav_files = list(subset_path.glob("*.wav"))
    if len(wav_files) == 0:
        logger.error("❌ No audio files found in subset")
        return False
    
    # Check metadata file
    metadata_file = subset_path / "ravdess_subset_metadata.csv"
    if not metadata_file.exists():
        logger.warning("⚠️ Metadata file not found")
    
    logger.info(f"✅ Setup verification passed")
    logger.info(f"   📁 Subset location: {subset_path}")
    logger.info(f"   📊 Audio files: {len(wav_files)}")
    logger.info(f"   📝 Metadata: {'✅' if metadata_file.exists() else '❌'}")
    
    return True


def create_sample_usage_script():
    """Create a sample script showing how to use the RAVDESS data."""
    sample_script = BACKEND_ROOT / "scripts" / "sample_ravdess_usage.py"
    
    sample_content = '''#!/usr/bin/env python3
"""
Sample RAVDESS Usage Script

Demonstrates how to load and use RAVDESS data in your application.
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
        print("❌ Metadata file not found. Please run setup first.")
        return None
    
    df = pd.read_csv(metadata_file)
    print(f"📊 Loaded {len(df)} audio samples")
    print(f"   Emotions: {df['emotion'].unique().tolist()}")
    print(f"   Actors: {df['actor'].nunique()} unique")
    print(f"   Gender balance: {df['gender'].value_counts().to_dict()}")
    
    return df

def load_audio_sample(filename: str, sr: int = 16000):
    """Load an audio sample from the subset."""
    audio_path = Path(__file__).parent.parent / "data" / "dev" / "ravdess_subset" / filename
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {filename}")
        return None, None
    
    # Load audio with librosa
    y, original_sr = librosa.load(audio_path, sr=sr)
    
    print(f"🎵 Loaded {filename}")
    print(f"   Duration: {len(y) / sr:.2f} seconds")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Shape: {y.shape}")
    
    return y, sr

def extract_basic_features(y, sr):
    """Extract basic audio features."""
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Root mean square energy
    rms = librosa.feature.rms(y=y)
    
    features = {
        'mfcc_mean': np.mean(mfccs, axis=1),
        'mfcc_std': np.std(mfccs, axis=1),
        'spectral_centroid_mean': np.mean(spectral_centroids),
        'spectral_centroid_std': np.std(spectral_centroids),
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms)
    }
    
    return features

def main():
    print("🎵 RAVDESS Sample Usage")
    print("=" * 50)
    
    # Load metadata
    df = load_subset_metadata()
    if df is None:
        return
    
    # Pick a random sample
    sample_row = df.sample(n=1).iloc[0]
    filename = sample_row['filename']
    emotion = sample_row['emotion']
    actor = sample_row['actor']
    gender = sample_row['gender']
    
    print(f"\\n🎯 Selected sample: {filename}")
    print(f"   Emotion: {emotion}")
    print(f"   Actor: {actor} ({gender})")
    
    # Load audio
    y, sr = load_audio_sample(filename)
    if y is None:
        return
    
    # Extract features
    features = extract_basic_features(y, sr)
    print(f"\\n📊 Extracted features:")
    for feature_name, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"   {feature_name}: {value.shape}")
        else:
            print(f"   {feature_name}: {value:.4f}")
    
    print("\\n✅ Sample usage completed!")

if __name__ == "__main__":
    main()
'''
    
    with open(sample_script, 'w') as f:
        f.write(sample_content)
    
    logger.info(f"📝 Created sample usage script: {sample_script}")


def main():
    parser = argparse.ArgumentParser(description="Set up RAVDESS dataset integration")
    parser.add_argument("--subset-only", action="store_true",
                       help="Only create subset (assume full dataset exists)")
    parser.add_argument("--subset-size", type=float, default=0.05,
                       help="Size of development subset (default: 0.05 = 5%%)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download and recreation")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip final verification")
    
    args = parser.parse_args()
    
    logger.info("🎵 RAVDESS Integration Setup")
    logger.info("=" * 50)
    
    try:
        # Check dependencies
        check_dependencies()
        
        # Download full dataset (unless subset-only)
        if not args.subset_only:
            if not run_download_script(force=args.force):
                logger.error("❌ Dataset download failed")
                sys.exit(1)
        
        # Create development subset
        if not run_subset_script(subset_size=args.subset_size, force=args.force):
            logger.error("❌ Subset creation failed") 
            sys.exit(1)
        
        # Verify setup
        if not args.no_verify:
            if not verify_setup():
                logger.error("❌ Setup verification failed")
                sys.exit(1)
        
        # Create sample usage script
        create_sample_usage_script()
        
        logger.info("🎉 RAVDESS setup completed successfully!")
        logger.info("🚀 Next steps:")
        logger.info("   1. Try: python scripts/sample_ravdess_usage.py")
        logger.info("   2. Set up database schema (Phase 2)")
        logger.info("   3. Implement preprocessing pipeline (Phase 3)")
        logger.info("   4. Create API endpoints (Phase 4)")
        
    except KeyboardInterrupt:
        logger.info("\\n⏹️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
