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

- **Frontend**: React 18 + TypeScript + Vite
- **UI Framework**: Tailwind CSS + shadcn/ui components
- **State Management**: TanStack Query
- **Data Visualization**: Custom React components with Chart.js integration
- **Audio Processing**: Web Audio API
- **Build Tool**: Vite with hot module replacement

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or bun package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AnasSAV/LIT-for-Voice.git
cd LIT-for-Voice/Frontend
```

3. **Set up the frontend**

   ```bash
   cd Frontend
   npm install && npm run dev
   ```

Start Redis server in Docker(in a new terminal)
docker compose up -d
```
3. **Set up the backend**

   ```cmd
   :: Create and activate virtual environment (Windows cmd)
   cd Backend
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip

   # Install dependencies
   pip install -r requirements.txt
   ```
   
```cmd
uvicorn app.main:app --reload
```

```cmd

### For Miniconda Users
:: 1️⃣ Initialize conda for your shell (Only if you have not used Conda on device before)
conda init cmd.exe

:: 2️⃣ Navigate to your project folder
cd Backend

:: 3️⃣ Create the environment with Python 3.10
conda create -n voice-lit python=3.10 -y

:: 4️⃣ Activate the environment
conda activate lit-voice

:: 5️⃣ Install web/backend dependencies
conda install -c pytorch -c nvidia -c conda-forge fastapi uvicorn starlette httpx python-multipart python-dotenv pydantic-settings anyio numpy pandas librosa pysoundfile transformers pytorch torchvision torchaudio pytorch-cuda=12.1 redis-py pytest pytest-asyncio requests -y

:: ✅ Environment 'lit-voice' is now ready for your project
uvicorn app.main:app --reload
```



4. Open your browser and navigate to `http://localhost:8080`


### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run build:dev` - Build for development
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

## Project Structure

```
src/
├── components/
│   ├── analysis/          # Perturbation and analysis tools
│   ├── audio/            # Audio-related components
│   ├── layout/           # Layout components
│   ├── panels/           # Main dashboard panels
│   ├── ui/               # Reusable UI components
│   └── visualization/    # Data visualization components
├── hooks/                # Custom React hooks
├── lib/                  # Utility functions
└── pages/                # Page components
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