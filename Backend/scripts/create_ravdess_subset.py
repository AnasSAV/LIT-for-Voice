#!/usr/bin/env python3
"""
RAVDESS Subset Creation Script

Creates a stratified subset of the RAVDESS dataset for development purposes.
This ensures faster iteration while maintaining representative samples across:
- All emotions (8 categories)
- Both genders (male/female)
- Both statements
- Multiple actors

Usage:
    python scripts/create_ravdess_subset.py [--size 0.05] [--force]
"""

import os
import sys
import shutil
import argparse
import hashlib
import json
import csv
from pathlib import Path
from typing import Dict, List, Set
import random
import logging
from collections import defaultdict

# Add the backend app to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the filename parser from download script
from scripts.download_ravdess import parse_ravdess_filename

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BACKEND_ROOT = Path(__file__).parent.parent
DATA_ROOT = BACKEND_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw" / "ravdess_full"
DEV_DATA_PATH = DATA_ROOT / "dev" / "ravdess_subset"
METADATA_FILE = DEV_DATA_PATH / "ravdess_subset_metadata.csv"
CHECKSUM_FILE = DEV_DATA_PATH / "ravdess_subset_checksums.json"


def calculate_file_checksum(filepath: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def analyze_dataset_distribution() -> Dict:
    """Analyze the full dataset to understand distribution."""
    if not RAW_DATA_PATH.exists():
        logger.error("❌ Raw dataset not found. Please run download_ravdess.py first.")
        sys.exit(1)
    
    audio_files = list(RAW_DATA_PATH.glob("**/*.wav"))
    logger.info(f"📊 Analyzing {len(audio_files)} audio files...")
    
    # Count by different categories
    emotion_counts = defaultdict(int)
    actor_counts = defaultdict(int)
    gender_counts = defaultdict(int)
    statement_counts = defaultdict(int)
    intensity_counts = defaultdict(int)
    
    file_metadata = []
    
    for audio_file in audio_files:
        metadata = parse_ravdess_filename(audio_file.name)
        if metadata:
            emotion_counts[metadata['emotion']] += 1
            actor_counts[metadata['actor']] += 1
            gender_counts[metadata['gender']] += 1
            statement_counts[metadata['statement']] += 1
            intensity_counts[metadata['intensity']] += 1
            
            file_metadata.append({
                'filepath': audio_file,
                'filename': audio_file.name,
                **metadata
            })
    
    logger.info("📈 Dataset Distribution:")
    logger.info(f"   Emotions: {dict(emotion_counts)}")
    logger.info(f"   Actors: {len(actor_counts)} unique ({dict(list(actor_counts.items())[:5])}...)")
    logger.info(f"   Gender: {dict(gender_counts)}")
    logger.info(f"   Statements: {len(statement_counts)} unique")
    logger.info(f"   Intensity: {dict(intensity_counts)}")
    
    return {
        'files': file_metadata,
        'distributions': {
            'emotions': dict(emotion_counts),
            'actors': dict(actor_counts),
            'genders': dict(gender_counts),
            'statements': dict(statement_counts),
            'intensities': dict(intensity_counts)
        }
    }


def create_stratified_subset(dataset_info: Dict, subset_fraction: float = 0.05) -> List[Dict]:
    """
    Create a stratified subset that maintains representation across key dimensions.
    
    Args:
        dataset_info: Full dataset analysis
        subset_fraction: Fraction of dataset to include (0.05 = 5%)
    
    Returns:
        List of selected file metadata
    """
    files = dataset_info['files']
    total_files = len(files)
    target_subset_size = max(50, int(total_files * subset_fraction))
    
    logger.info(f"🎯 Creating stratified subset:")
    logger.info(f"   Total files: {total_files}")
    logger.info(f"   Subset fraction: {subset_fraction} ({subset_fraction*100:.1f}%)")
    logger.info(f"   Target subset size: {target_subset_size}")
    
    # Group files by key characteristics
    emotion_groups = defaultdict(list)
    actor_groups = defaultdict(list)
    
    for file_info in files:
        emotion_groups[file_info['emotion']].append(file_info)
        actor_groups[file_info['actor']].append(file_info)
    
    selected_files = []
    used_files = set()
    
    # Strategy: Select files to ensure representation across emotions and actors
    emotions = list(emotion_groups.keys())
    
    # Calculate how many files per emotion
    files_per_emotion = target_subset_size // len(emotions)
    remaining_files = target_subset_size % len(emotions)
    
    logger.info(f"📋 Selection strategy: ~{files_per_emotion} files per emotion")
    
    for i, emotion in enumerate(emotions):
        emotion_files = emotion_groups[emotion]
        # Add extra file to some emotions if there's remainder
        emotion_target = files_per_emotion + (1 if i < remaining_files else 0)
        
        # Within this emotion, try to get diverse actors and statements
        selected_emotion_files = []
        actors_in_emotion = defaultdict(list)
        
        for file_info in emotion_files:
            actors_in_emotion[file_info['actor']].append(file_info)
        
        # Sample from different actors
        actor_list = list(actors_in_emotion.keys())
        random.shuffle(actor_list)
        
        files_selected = 0
        actor_idx = 0
        
        while files_selected < emotion_target and files_selected < len(emotion_files):
            if actor_idx >= len(actor_list):
                actor_idx = 0  # Wrap around
            
            actor = actor_list[actor_idx]
            actor_files = actors_in_emotion[actor]
            
            # Select a random file from this actor for this emotion
            available_files = [f for f in actor_files if f['filename'] not in used_files]
            
            if available_files:
                selected_file = random.choice(available_files)
                selected_emotion_files.append(selected_file)
                used_files.add(selected_file['filename'])
                files_selected += 1
            
            actor_idx += 1
        
        selected_files.extend(selected_emotion_files)
        logger.info(f"   {emotion}: selected {len(selected_emotion_files)} files")
    
    logger.info(f"✅ Final subset size: {len(selected_files)} files")
    
    # Verify diversity
    final_emotions = defaultdict(int)
    final_actors = set()
    final_genders = defaultdict(int)
    
    for file_info in selected_files:
        final_emotions[file_info['emotion']] += 1
        final_actors.add(file_info['actor'])
        final_genders[file_info['gender']] += 1
    
    logger.info("🔍 Subset diversity check:")
    logger.info(f"   Emotions: {dict(final_emotions)}")
    logger.info(f"   Unique actors: {len(final_actors)}")
    logger.info(f"   Gender balance: {dict(final_genders)}")
    
    return selected_files


def copy_subset_files(selected_files: List[Dict], force: bool = False):
    """Copy selected files to the development subset directory."""
    
    if DEV_DATA_PATH.exists() and not force:
        logger.info(f"✅ Subset directory already exists: {DEV_DATA_PATH}")
        existing_files = list(DEV_DATA_PATH.glob("*.wav"))
        if len(existing_files) > 0:
            logger.info(f"   Found {len(existing_files)} existing files")
            return existing_files
    
    # Create subset directory
    DEV_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📂 Creating subset in: {DEV_DATA_PATH}")
    
    copied_files = []
    checksums = {}
    
    for i, file_info in enumerate(selected_files, 1):
        src_path = file_info['filepath']
        dst_path = DEV_DATA_PATH / file_info['filename']
        
        logger.info(f"   [{i}/{len(selected_files)}] {file_info['filename']}")
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        
        # Calculate checksum for integrity
        checksum = calculate_file_checksum(dst_path)
        checksums[file_info['filename']] = {
            'md5': checksum,
            'size': dst_path.stat().st_size,
            'emotion': file_info['emotion'],
            'actor': file_info['actor'],
            'gender': file_info['gender']
        }
        
        copied_files.append(dst_path)
    
    # Save checksums for verification
    with open(CHECKSUM_FILE, 'w') as f:
        json.dump(checksums, f, indent=2)
    
    logger.info(f"✅ Copied {len(copied_files)} files to development subset")
    logger.info(f"📝 Checksums saved to: {CHECKSUM_FILE}")
    
    return copied_files


def save_subset_metadata(selected_files: List[Dict]):
    """Save metadata for the subset files as CSV."""
    
    logger.info(f"💾 Saving metadata to: {METADATA_FILE}")
    
    fieldnames = [
        'filename', 'emotion', 'intensity', 'statement', 'repetition', 
        'actor', 'gender', 'modality', 'vocal_channel'
    ]
    
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file_info in selected_files:
            row = {field: file_info.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    logger.info("✅ Metadata saved successfully")


def verify_subset_integrity() -> bool:
    """Verify the integrity of the created subset."""
    if not DEV_DATA_PATH.exists():
        logger.error("❌ Subset directory does not exist")
        return False
    
    if not CHECKSUM_FILE.exists():
        logger.warning("⚠️ No checksum file found, skipping integrity check")
        return True
    
    with open(CHECKSUM_FILE, 'r') as f:
        stored_checksums = json.load(f)
    
    logger.info("🔍 Verifying subset integrity...")
    
    verification_passed = True
    for filename, stored_info in stored_checksums.items():
        file_path = DEV_DATA_PATH / filename
        
        if not file_path.exists():
            logger.error(f"❌ Missing file: {filename}")
            verification_passed = False
            continue
        
        current_checksum = calculate_file_checksum(file_path)
        if current_checksum != stored_info['md5']:
            logger.error(f"❌ Checksum mismatch for {filename}")
            verification_passed = False
        
        current_size = file_path.stat().st_size
        if current_size != stored_info['size']:
            logger.error(f"❌ Size mismatch for {filename}")
            verification_passed = False
    
    if verification_passed:
        logger.info("✅ Subset integrity verification passed")
    else:
        logger.error("❌ Subset integrity verification failed")
    
    return verification_passed


def main():
    parser = argparse.ArgumentParser(description="Create RAVDESS development subset")
    parser.add_argument("--size", type=float, default=0.05,
                       help="Fraction of dataset to include (default: 0.05 = 5%)")
    parser.add_argument("--force", action="store_true",
                       help="Force recreation even if subset exists")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing subset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible subsets")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    logger.info("🎭 RAVDESS Subset Creation")
    logger.info("=" * 50)
    
    if args.verify_only:
        success = verify_subset_integrity()
        sys.exit(0 if success else 1)
    
    # Analyze full dataset
    dataset_info = analyze_dataset_distribution()
    
    # Create stratified subset
    selected_files = create_stratified_subset(dataset_info, args.size)
    
    # Copy files
    copied_files = copy_subset_files(selected_files, force=args.force)
    
    # Save metadata
    save_subset_metadata(selected_files)
    
    # Verify integrity
    verify_subset_integrity()
    
    logger.info("✅ RAVDESS subset creation completed!")
    logger.info(f"📁 Subset location: {DEV_DATA_PATH}")
    logger.info(f"📊 Subset size: {len(selected_files)} files")
    logger.info("🚀 Next steps:")
    logger.info("   1. Set up database schema")
    logger.info("   2. Run preprocessing pipeline")
    logger.info("   3. Implement API endpoints")


if __name__ == "__main__":
    main()
