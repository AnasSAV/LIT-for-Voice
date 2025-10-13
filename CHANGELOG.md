# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-10-13

### Added
- **Initial stable release of LIT for Voice** - A comprehensive Learning Interpretability Tool for Audio Models
- **Frontend Application** (React 18 + TypeScript + Vite)
  - Interactive audio waveform visualization with playback controls
  - Model prediction analysis dashboard
  - Attention pattern visualization for transformer-based audio models
  - High-dimensional audio embedding visualization in 2D/3D space
  - Gradient-based saliency mapping for audio inputs
  - Comprehensive perturbation tools for model robustness testing
  - Responsive UI built with Tailwind CSS and shadcn/ui components
  - Audio file upload and management system
  
- **Backend API** (FastAPI + Python 3.11)
  - RESTful API for audio processing and model inference
  - Support for transformer-based audio models (Whisper, Wav2Vec2)
  - Redis caching for predictions and analysis results
  - Audio perturbation service with various transformation techniques
  - Custom dataset handling and management
  - Session-based file management
  - Comprehensive test suite with pytest
  
- **Core Features**
  - Audio data management with metadata support
  - Interactive waveform viewer with zoom and navigation
  - Model prediction analysis with confidence scores
  - Attention mechanism visualization
  - Embedding space exploration tools
  - Saliency map generation for interpretability
  - Multiple audio perturbation techniques
  - Real-time model inference and analysis
  
- **Infrastructure**
  - Docker Compose setup for Redis
  - Comprehensive development environment setup
  - CI/CD ready project structure
  - Detailed documentation and setup guides
  
- **Documentation**
  - Comprehensive README with installation instructions
  - Contributing guidelines for developers
  - Code of conduct for community participation
  - Security policy for responsible disclosure
  - Project structure documentation
  
### Technical Stack
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query
- **Backend**: FastAPI, Python 3.11, Redis, PyTorch, Transformers, Librosa
- **Audio Processing**: Web Audio API, Librosa, SoundFile
- **Visualization**: Custom React components, Chart.js integration
- **Development**: ESLint, Prettier, pytest, Docker

### Supported Models
- OpenAI Whisper (speech recognition)
- Facebook Wav2Vec2 (speech representation learning)
- Custom transformer-based audio models

### Sample Datasets
- Common Voice validation subset
- RAVDESS emotion recognition subset
- Support for custom audio dataset uploads

---

**Full Changelog**: https://github.com/AnasSAV/LIT-for-Voice/commits/v1.0.0