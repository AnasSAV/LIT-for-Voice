# LIT for Voice - Learning Interpretability Tool for Audio Models

## Introduction

Interpreting how deep learning models make decisions is crucial, especially in high-stakes applications like speech recognition, emotion detection, and speaker identification. While the Learning Interpretability Tool (LIT) enables exploration of text and tabular models, there's a lack of equivalent tools for voice-based models. Voice data poses additional challenges due to its temporal nature and multi-modal representations (e.g., waveform, spectrogram). 

This project aims to extend the interpretability paradigm to audio, empowering researchers and developers to analyze and debug speech models with greater transparency. LIT for Voice provides an interactive web-based interface for exploring audio models through various visualization techniques, attention mechanisms, and perturbation analyses.

## Features

- **Audio Data Management**: Upload and manage audio datasets with metadata
- **Waveform Visualization**: Interactive waveform viewer with playback controls
- **Model Prediction Analysis**: Examine model predictions and confidence scores
- **Attention Visualization**: Explore attention patterns in transformer-based audio models
- **Embedding Analysis**: Visualize high-dimensional audio embeddings in 2D/3D space
- **Saliency Mapping**: Identify important regions in audio input using gradient-based methods
- **Perturbation Tools**: Apply various audio perturbations to test model robustness
- **Interactive Dashboard**: Comprehensive interface for exploring model behavior

## Tech Stack

**Frontend:**
- React 18 + TypeScript + Vite
- Tailwind CSS + shadcn/ui components (Radix UI)
- TanStack Query for state management and API calls
- Plotly.js and Recharts for data visualization
- WaveSurfer.js for audio waveform visualization
- React Router for navigation
- Vite with hot module replacement

**Backend:**
- FastAPI + Python
- Redis for session management and queue operations
- PyTorch for ML model inference
- Librosa for audio processing
- Pydantic for data validation
- pytest for testing with fakeredis

## Getting Started

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (3.8 or higher)
- **Docker Desktop** (for Redis)
- npm or bun package manager
- pip (Python package manager)

### Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/AnasSAV/LIT-for-Voice.git
cd LIT-for-Voice
```

2. **Start Redis (Required for Backend):**
```bash
cd Backend
docker-compose up
```
Keep this terminal open. Redis will run on port 6379.

3. **Setup and Start Backend (New Terminal):**
```bash
cd Backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be available at `http://localhost:8000`

4. **Setup and Start Frontend (New Terminal):**
```bash
cd Frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```
Frontend will be available at `http://localhost:8080`

5. **Verify Setup:**
   - Frontend: `http://localhost:8080`
   - Backend API Docs (Swagger): `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/health`
   - Datasets: `http://localhost:8000/datasets`
   - Active Dataset: `http://localhost:8000/datasets/active`
   - List Dataset Files: `http://localhost:8000/datasets/files?limit=10&offset=0`
   - Get Dataset File: `http://localhost:8000/datasets/file?relpath=REL_PATH[&id=ravdess_subset]`

### Available Scripts

**Frontend:**
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run build:dev` - Build for development
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

**Backend:**
- `uvicorn app.main:app --reload` - Start development server with auto-reload
- `pytest` - Run all tests
- `pytest -v` - Run tests with verbose output
- `docker-compose up` - Start Redis service

## Project Structure

```
LIT-for-Voice/
├── Frontend/                    # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── analysis/        # Perturbation and analysis tools
│   │   │   ├── audio/          # Audio-related components
│   │   │   ├── layout/         # Layout components (MainLayout, Toolbar)
│   │   │   ├── panels/         # Main dashboard panels
│   │   │   ├── ui/             # Reusable shadcn/ui components
│   │   │   └── visualization/  # Data visualization components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Utility functions
│   │   └── pages/              # Page components
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
├── Backend/                     # FastAPI Python backend
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/         # API route handlers
│   │   │       ├── session.py   # Session management
│   │   │       ├── results.py   # Results caching
│   │   │       ├── health.py    # Health checks
│   │   │       └── datasets.py  # Dataset browsing & file serving
│   │   ├── core/               # Core functionality
│   │   │   ├── redis.py        # Redis client
│   │   │   ├── session.py      # Session middleware
│   │   │   └── settings.py     # App settings
│   │   ├── services/           # Business logic
│   │   │   └── queue_service.py # Queue management
│   │   └── main.py             # FastAPI application
│   ├── tests/                  # Backend tests
│   ├── requirements.txt
│   ├── docker-compose.yml
│   └── pytest.ini
├── README.md
```

## Usage

1. **Upload Audio Data**: Use the audio uploader to load your audio files and associated metadata
2. **Model Integration**: Connect your audio model to generate predictions and embeddings
3. **Explore Visualizations**: Navigate through different panels to explore:
   - Waveform representations
   - Model predictions and confidence scores
   - Attention patterns
   - Embedding clusters
   - Saliency maps
4. **Apply Perturbations**: Test model robustness using various audio perturbation techniques
5. **Analyze Results**: Use the interactive dashboard to gain insights into model behavior

## Development Status

**Current State:**
- ✅ Frontend UI components and dashboard layout
- ✅ Backend API structure with FastAPI
- ✅ Redis integration for session management
- ✅ Testing framework with pytest
- ✅ Dataset endpoints for browsing audio (list/active/files/file) when data is present under `Backend/data/`
- ⚠️ Frontend-Backend integration (partial)
- ❌ Audio upload API and end-to-end processing pipeline
- ❌ ML model integration
- ❌ Audio visualization features

**Note**: The project is under active development. The UI is functional but backend integration for audio processing is incomplete.

## Troubleshooting

### Windows PowerShell Issues
If you encounter "execution of scripts is disabled" errors:
1. Run PowerShell as Administrator
2. Execute: `Set-ExecutionPolicy RemoteSigned`
3. Alternative: Use Node directly: `& "$env:ProgramFiles\nodejs\node.exe" ./node_modules/vite/bin/vite.js`

### Docker Issues
- Ensure Docker Desktop is running before starting Redis
- If port 6379 is in use, modify the port in `docker-compose.yml`

### Port Conflicts
- Frontend runs on port 8080
- Backend runs on port 8000
- Redis runs on port 6379
Ensure these ports are available or modify the configuration files.

## Configuration

- **Backend settings** come from `Backend/app/core/settings.py` via Pydantic BaseSettings. Set these as environment variables if you need to override defaults:
  - `REDIS_URL` (default: `redis://localhost:6379/0`)
  - `SESSION_COOKIE_NAME` (default: `sid`)
  - `SESSION_TTL_SECONDS` (default: `86400`)
  - `COOKIE_SECURE` (default: `False`)
  - `COOKIE_SAMESITE` (default: `lax`)
  - `COOKIE_DOMAIN` (default: empty)

- **Frontend API base URL** is defined in `Frontend/src/lib/api/datasets.ts`:
  - Defaults to `http://localhost:8000`.
  - Optional override using Vite env: set `VITE_API_BASE`, e.g. create `Frontend/.env` with:
    ```
    VITE_API_BASE=http://localhost:8000
    ```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

- **Anas Hussaindeen** - [GitHub Profile](https://github.com/AnasSAV)
- **Chandupa Ambepitiya** - [GitHub Profile](https://github.com/Chand2103)
- **Dewmike Amarasinghe** - [GitHub Profile](https://github.com/DewmikeAmarasinghe)

## Acknowledgments

- Inspired by Google's Learning Interpretability Tool (LIT)
- Built with modern React ecosystem and TypeScript
- Special thanks to the open-source community for the amazing tools and libraries

## Roadmap

- [ ] Backend API integration for model serving
- [ ] Support for more audio model architectures
- [ ] Advanced perturbation techniques
- [ ] Real-time audio processing capabilities
- [ ] Export functionality for visualizations
- [ ] Multi-language support
- [ ] Plugin system for custom analysis tools

**Note**: This project is under active development. Features and API may change between versions.