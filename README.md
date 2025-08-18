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
- pytest for testing

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

## How to Run (Windows)

The app has a FastAPI backend and a React (Vite) frontend. Run the backend first, then the frontend.

### 1) Backend (FastAPI)

Run these in a terminal from the `Backend/` directory.

1. Create and activate a virtual environment

```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

2. Install dependencies

```cmd
pip install -r requirements.txt
```

Note: `torch` is a large dependency. Ensure your Python version is supported and you have sufficient disk space.

3. Start Redis

- Using Docker (recommended): ensure Docker Desktop is running, then from `Backend/`:

```cmd
docker compose up -d
```

This starts Redis 7 on `localhost:6379` matching the default `REDIS_URL` in `Backend/app/core/settings.py`.

- Or run your own Redis service and set `REDIS_URL` if it differs:

```cmd
set REDIS_URL=redis://<host>:<port>/0
```

4. Start the API server

```cmd
set PYTHONPATH=.
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints to verify:

```cmd
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/datasets
curl http://localhost:8000/datasets/summary
```

Datasets are discovered under `Backend/data/`. If you already downloaded them there, the `/datasets` endpoints will reflect them.

### 2) Frontend (Vite + React)

Run these from the `Frontend/` directory.

1. Install Node dependencies

```cmd
npm install
```

2. Configure API base (optional)

If your backend is not at `http://localhost:8000`, set `Frontend/.env`:

```ini
VITE_API_BASE=http://<your-backend-host>:<port>
```

3. Start the dev server

```cmd
npm run dev
```

Vite will serve the app at `http://localhost:8080`. The backend CORS in `Backend/app/main.py` already allows `http://localhost:8080` and `http://localhost:5173`.

### 3) Run tests (optional)

From `Backend/` with the venv active:

```cmd
pytest -q
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

**Note**: The project is under active development. Features and API may change between versions.

## Configuration

- **Backend settings** come from `Backend/app/core/settings.py` via Pydantic BaseSettings. Set these as environment variables if you need to override defaults:

  - `REDIS_URL` (default: `redis://localhost:6379/0`)
  - `SESSION_COOKIE_NAME` (default: `sid`)
  - `SESSION_TTL_SECONDS` (default: `86400`)
  - `DATASET_CACHE_TTL_SECONDS` (default: `86400`)
  - `COOKIE_SECURE` (default: `False`)
  - `COOKIE_SAMESITE` (default: `lax`)
  - `COOKIE_DOMAIN` (default: empty)
  

- **Frontend API base URL** is defined in `Frontend/src/lib/api/datasets.ts`:
  - Defaults to `http://localhost:8000`.
  - Optional override using Vite env: set `VITE_API_BASE`, e.g. create `Frontend/.env` with:
    ```
    VITE_API_BASE=http://localhost:8000
    ```

## Dataset manifest caching

The backend caches static dataset metadata in Redis to speed up browsing:

- Redis keys per dataset id:
  - `dataset:{id}:manifest` – JSON list with per-file entries: `id`, `relpath`, `filename`, `size`, `duration`, `label`, `h` (stable hash)
  - `dataset:{id}:summary` – JSON summary: `total`, `total_bytes`, and `label_counts`
  - `dataset:{id}:version` – snapshot hash of directory contents used for invalidation
- Cache expiration: controlled by `DATASET_CACHE_TTL_SECONDS`.
- Versioning: if the on-disk version differs, the manifest is rebuilt automatically.

Related endpoints:

- `GET /datasets/files?limit=&offset=` – paginated view backed by the cached manifest
- `GET /datasets/summary` – returns the cached summary for the active (or specified) dataset
- `POST /datasets/reindex` – force a rebuild; body: `{ "id": "ravdess_subset" | "ravdess_full" | "common_voice_en" }`

Results caching:

- `GET /results/{model}/{h}` and `POST /results/{model}/{h}` operate on a per-sample cache.
- `POST /results/{model}/batch` – fetch multiple results in one call by providing an array of `h` values.

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