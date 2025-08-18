# RAVDESS Dataset Integration Scripts

This directory contains scripts for integrating the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset into the LIT-for-Voice project.

## 📁 Scripts Overview

### 📋 Script Details

#### `setup_ravdess.py` - **Main Setup Script**
One-stop script that handles the complete RAVDESS integration setup.

**Options:**
- `--subset-only`: Skip download, only create subset (assumes full dataset exists)
- `--subset-size 0.05`: Size of development subset (5% by default)
- `--force`: Force re-download and recreation
- `--no-verify`: Skip final verification

**What it does:**
1. ✅ Checks and installs dependencies (kaggle, librosa, pandas, numpy)
2. 📥 Downloads RAVDESS dataset from Kaggle 
3. 🎭 Creates stratified development subset
4. 🔍 Verifies setup integrity
5. 📝 Creates sample usage script

---

#### `download_ravdess.py` - **Dataset Downloader**
Downloads the full RAVDESS dataset from Kaggle.

**Features:**
- ✅ Kaggle API authentication check
- 📥 Downloads to `data/raw/ravdess_full/`
- 🔍 Verifies download integrity
- 📝 Creates .gitignore for data directories
- 🎯 Parses RAVDESS filename format

---

#### `create_ravdess_subset.py` - **Subset Creator**
Creates a stratified subset of RAVDESS for faster development iteration.

**Options:**
- `--size 0.05`: Fraction of dataset (5% = ~72 files from 1,440)
- `--force`: Force recreation even if subset exists
- `--seed 42`: Random seed for reproducible subsets
- `--verify-only`: Only verify existing subset

**Stratification Strategy:**
- 🎭 Maintains representation across all 8 emotions
- 👥 Includes diverse actors (male/female balance)
- 📝 Preserves both statement types
- 🎯 Ensures variety in intensity levels

**Output:**
- Audio files: `data/dev/ravdess_subset/*.wav`
- Metadata: `ravdess_subset_metadata.csv`
- Checksums: `ravdess_subset_checksums.json`

---

#### `sample_ravdess_usage.py` - **Usage Example**
Demonstrates how to load and process RAVDESS data in your application.

**Demonstrates:**
- 📊 Loading metadata from CSV
- 🎵 Audio loading with librosa
- 📈 Basic feature extraction (MFCC, spectral centroid, ZCR, RMS)
- 🔍 Dataset exploration and analysis

## 🗂️ Data Structure

After running the scripts, your data directory will look like:

```
Backend/data/
├── .gitignore                    # Git ignore for audio files
├── raw/
│   └── ravdess_full/            # Full dataset (~1,440 files, ~5GB)
├── dev/
│   └── ravdess_subset/          # Development subset (~72 files, ~250MB)
│       ├── *.wav                # Audio files
│       ├── ravdess_subset_metadata.csv
│       └── ravdess_subset_checksums.json
└── processed/                   # For future processed data
```

## 🎵 RAVDESS Dataset Details

### Filename Format
`Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav`

**Example:** `03-01-06-01-02-01-12.wav`
- `03`: Audio-only
- `01`: Speech  
- `06`: Fearful
- `01`: Normal intensity
- `02`: "Dogs are sitting by the door"
- `01`: First repetition
- `12`: Actor 12 (female)

### Metadata Fields
- **Emotions**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Actors**: 24 professional actors (12 male, 12 female)
- **Statements**: 2 different sentences
- **Intensities**: normal, strong
- **Total**: 1,440 audio files

## 🔗 Integration with LIT-for-Voice

This Phase 1 setup prepares the foundation for:

**Phase 2**: Database schema and migrations
**Phase 3**: Audio preprocessing pipeline  
**Phase 4**: FastAPI endpoints for dataset management
**Phase 5**: Frontend integration with AudioDataTable
**Phase 6**: Testing and documentation

The stratified subset ensures fast development iteration while maintaining representative samples across all emotional categories and demographic groups.
