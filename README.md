# 🎤 LIT for Voice - Learning Interpretability Tool for Audio Models

<div align="center">
  <p>
    <a href="#">
      <img width="300" src="Frontend/public/lit_web.png" alt="LIT for Voice">
    </a>
  </p>
  <p>
    <a href="#">
      <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen" alt="Node.js">
    </a>
  </p>
</div>

## 🎯 Introduction

LIT for Voice is an open-source toolkit designed to bring interpretability to audio machine learning models. In the rapidly evolving field of speech and audio processing, understanding model decisions is as crucial as model performance. This tool provides an interactive, web-based interface to explore, analyze, and interpret audio models with unprecedented clarity.

### Why LIT for Voice?

- **Temporal Understanding**: Audio data's sequential nature requires specialized visualization and analysis tools
- **Multi-modal Analysis**: Simultaneously view waveforms, spectrograms, and model internals
- **Model Debugging**: Identify failure modes and biases in audio models
- **Research Acceleration**: Speed up model development with intuitive visual feedback

## ✨ Features

### 🔍 Model Analysis

- **Prediction Inspection**: Dive into model outputs with confidence scores and alternative predictions
- **Attention Visualization**: Explore self-attention patterns in transformer architectures
- **Saliency Maps**: Identify which parts of the audio most influence model decisions

### 🎧 Audio Tools

- **Interactive Waveform**: Zoom, pan, and play audio with synchronized visualizations
- **Multi-format Support**: Load WAV, MP3, and other common audio formats
- **Audio Augmentation**: Apply various transformations to test model robustness

### 📊 Visualization

- **Embedding Projections**: Visualize high-dimensional audio representations in 2D/3D space
- **Time-aligned Views**: Correlate model behavior with specific audio segments
- **Custom Layouts**: Arrange and save analysis views for different workflows

### 🛠️ Technical Features

- **Model-Agnostic**: Works with any PyTorch-based audio model
- **Real-time Interaction**: Immediate feedback on model behavior
- **Extensible Architecture**: Easily add new analysis methods and visualizations

## 🚀 Tech Stack

### Frontend

| Technology               | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| React 18 + TypeScript    | Core UI framework with type safety        |
| Vite                     | Fast development server and build tool    |
| Tailwind CSS + shadcn/ui | Styling and accessible components         |
| TanStack Query           | Server state management and data fetching |
| WaveSurfer.js            | Audio waveform visualization and playback |
| Plotly.js & Recharts     | Interactive data visualizations           |
| React Router             | Client-side routing                       |
| Zod                      | Runtime type validation                   |

### Backend

| Technology | Purpose                                 |
| ---------- | --------------------------------------- |
| FastAPI    | High-performance API framework          |
| PyTorch    | Deep learning model inference           |
| Librosa    | Audio feature extraction and processing |
| Redis      | Caching and job queue                   |
| Pydantic   | Data validation and settings management |
| SQLAlchemy | Database ORM                            |
| pytest     | Testing framework                       |

## 📁 Project Structure

```
LIT-for-Voice/
├── Frontend/                    # React TypeScript frontend
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   │   ├── analysis/       # Model analysis tools
│   │   │   ├── audio/          # Audio visualization components
│   │   │   ├── layout/         # Layout components
│   │   │   ├── panels/         # Main application panels
│   │   │   └── ui/             # Base UI components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── lib/                # Utility functions
│   │   │   └── api/            # API client and types
│   │   └── pages/              # Application routes
│   └── public/                 # Static assets
│
├── Backend/                     # FastAPI backend
│   ├── app/
│   │   ├── api/                # API endpoints
│   │   ├── core/               # Core application logic
│   │   ├── models/             # Database models
│   │   ├── services/           # Business logic
│   │   └── utils/              # Utility functions
│   ├── data/                   # Dataset storage
│   ├── tests/                  # Test suite
│   └── scripts/                # Utility scripts
│
└── docs/                       # Documentation
    ├── api/                    # API documentation
    └── guides/                 # User and developer guides
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Redis
- FFmpeg

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/LIT-for-Voice.git
   cd LIT-for-Voice
   ```

2. **Set up the backend**

   ```cmd
   :: Create and activate virtual environment (Windows cmd)
   cd Backend
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up the frontend**

   ```bash
   cd Frontend
   npm install
   ```

4. **Optional: Configure environment variables (.env)**
   The backend reads environment variables and a local `.env` (via Pydantic Settings). You generally don't need a `.env` for local dev. To override defaults, create `Backend/.env` with, for example:
   ```env
   # Backend/.env
   REDIS_URL=redis://localhost:6379/0
   ```

### Running the Application

1. **Start Redis**

   ```bash
   # Using Docker Compose (recommended) - run from repo root
   docker compose up -d

   # Or start your local Redis service manually
   # Linux/macOS:
   #   redis-server
   # Windows (service/WSL/Docker Desktop):
   #   Use Docker Compose above or install Redis for Windows
   ```

2. **Start the backend server**

   ```bash
   cd Backend
   uvicorn app.main:app --reload
   ```

3. **Start the frontend development server**

   ```bash
   cd Frontend
   npm run dev
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:8080`

### Docker: Redis only

A minimal `docker-compose.yml` is provided at the repo root to run Redis with RDB snapshots (AOF disabled):

```bash
# Start Redis (detached)
docker compose up -d

# Stop services
docker compose down

# Stop and remove data volume
docker compose down -v
```

Set `REDIS_URL` accordingly:

- Backend outside Docker: `REDIS_URL=redis://localhost:6379/0`
- If you later add the backend as a Compose service: `redis://redis:6379/0`

## 📚 Documentation

For detailed documentation, please visit our [documentation site](https://your-docs-site.com).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or feedback, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).


## How to Run (Windows)

The app has a FastAPI backend and a React (Vite) frontend. Run the backend first, then the frontend.

### 1) Backend (FastAPI)

Run these in a terminal from the `Backend/` directory.

1. Create and activate a virtual environment

```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
````

2. Install dependencies

```cmd
pip install -r requirements.txt
```

Note: `torch` is a large dependency. Ensure your Python version is supported and you have sufficient disk space.

3. Start Redis

- Using Docker (recommended): ensure Docker Desktop is running, then from the repository root:

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

Vite will serve the app at `http://localhost:5173` by default. The backend CORS in `Backend/app/main.py` already allows `http://localhost:5173`.

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

## API Endpoints and Caching

For a deeper dive into cache internals and Redis keys, see `Backend/docs/README.md`.

### Dataset Browsing

#### List Dataset Files

- `GET /datasets/files`
  - Query Params:
    - `limit`: Number of files to return (default: 50)
    - `offset`: Pagination offset (default: 0)
  - Response:
    ```json
    {
      "files": [
        {
          "id": "unique_id",
          "filename": "audio.wav",
          "relpath": "path/to/audio.wav",
          "size": 12345,
          "duration": 1.23,
          "label": "emotion_label",
          "h": "unique_hash",
          "meta": {
            "emotion": "happy",
            "actor": "16",
            "gender": "female"
          }
        }
      ],
      "total": 100,
      "active": "ravdess_subset"
    }
    ```
  - Uses Redis caching with automatic invalidation when dataset changes

#### Dataset Summary

- `GET /datasets/summary`
  - Returns dataset statistics
  - Response:
    ```json
    {
      "total": 100,
      "total_bytes": 12345678,
      "label_counts": { "happy": 30, "sad": 20, "neutral": 50 },
      "meta_constants": { "modality": "audio-only", "vocal_channel": "speech" }
    }
    ```

#### Rebuild Dataset Index

- `POST /datasets/reindex`
  - Body: `{"id": "dataset_id"}`
  - Forces a rebuild of the dataset manifest
  - Returns: `{"ok": true}` on success

### Results Caching

#### Get Single Result

- `GET /results/{model}/{h}`
  - Fetches a single cached result
  - `h` is the unique hash from the dataset file entry

#### Batch Results

- `POST /results/{model}/batch`
  - Body: `{"hashes": ["h1", "h2", ...]}`
  - Fetches multiple results in one request
  - Returns: `{"ok": true, "payloads": {"h1": {...}, "h2": {...}}}`

### Caching Behavior

- **Manifest Caching**:

  - Redis keys:
    - `dataset:{id}:manifest`: File metadata
    - `dataset:{id}:summary`: Dataset statistics
    - `dataset:{id}:version`: Content hash for invalidation
  - Auto-rebuilds when files change
  - TTL controlled by `DATASET_CACHE_TTL_SECONDS`

- **Result Caching**:
  - Key: `result:{model}:{h}`
  - Persistent until explicitly cleared or expired
  - Batch API reduces round-trips for multiple predictions

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
