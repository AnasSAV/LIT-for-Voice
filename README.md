<p align="center">
  <img width="1920" height="1080" alt="LIT for Voice" src="https://github.com/user-attachments/assets/42155dcb-2526-4780-ab5c-8374777a4144" />
</p>

# LIT for Voice

<p align="center">
  <a href="https://github.com/AnasSAV/LIT-for-Voice">
    <img src="https://img.shields.io/badge/version-v1.0-blue" alt="Version"/>
  </a>
  <a href="https://github.com/AnasSAV/LIT-for-Voice/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/AnasSAV/LIT-for-Voice" alt="License"/>
  </a>
  <a href="https://github.com/AnasSAV/LIT-for-Voice/stargazers">
    <img src="https://img.shields.io/github/stars/AnasSAV/LIT-for-Voice" alt="Stars"/>
  </a>
  <a href="https://github.com/AnasSAV/LIT-for-Voice/network/members">
    <img src="https://img.shields.io/github/forks/AnasSAV/LIT-for-Voice" alt="Forks"/>
  </a>
  <a href="https://github.com/AnasSAV/LIT-for-Voice/issues">
    <img src="https://img.shields.io/github/issues/AnasSAV/LIT-for-Voice" alt="Issues"/>
  </a>
</p>

> **Learning Interpretability Tool for Audio Models**

Interpreting how deep learning models make decisions is crucial, especially in high-stakes applications like speech recognition, emotion detection, and speaker identification. While the Learning Interpretability Tool (LIT) enables exploration of text and tabular models, there's a lack of equivalent tools for voice-based models. Voice data poses additional challenges due to its temporal nature and multi-modal representations (e.g., waveform, spectrogram).

LIT for Voice extends the interpretability paradigm to audio models, providing researchers and developers with tools to analyze and debug speech models with greater transparency. Through interactive visualizations, attention mechanisms, and perturbation analyses, you can gain deeper insights into how your audio models make decisions.

## 🚀 Features

* 🔊 **Audio Data Management**: Upload and manage audio datasets with metadata
* 📊 **Waveform Visualization**: Interactive waveform viewer with playback controls
* 🧠 **Model Prediction Analysis**: Examine model predictions and confidence scores
* 👀 **Attention Visualization**: Explore attention patterns in transformer-based audio models
* 📈 **Embedding Analysis**: Visualize high-dimensional audio embeddings in 2D/3D space
* 🔍 **Saliency Mapping**: Identify important regions in audio input using gradient-based methods
* 🔄 **Perturbation Tools**: Apply various audio perturbations to test model robustness
* 💻 **Interactive Dashboard**: Comprehensive interface for exploring model behavior

## 🧰 Tech Stack

* **Frontend**: React 18 + TypeScript + Vite
* **UI Framework**: Tailwind CSS + shadcn/ui components
* **State Management**: TanStack Query
* **Data Visualization**: Custom React components with Chart.js integration
* **Audio Processing**: Web Audio API
* **Backend**: FastAPI + Python 3.11
* **Models**: Transformer-based audio models (Whisper, Wav2Vec2)
* **Storage**: Redis for caching predictions and results

## 📋 Prerequisites

- **Frontend**:
  - Node.js (v18 or higher)
  - npm or bun package manager

- **Backend**:
  - Python 3.11
  - Docker (for Redis)

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/AnasSAV/LIT-for-Voice.git
cd LIT-for-Voice
```

### 2. Set up the Frontend
```bash
cd Frontend
npm install
npm run dev
```

### 3. Start Redis server in Docker
```bash
# In a new terminal
cd Backend
docker compose up -d
```

### 4. Set up the Backend

#### Using Python venv (recommended)
```bash
cd Backend
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix or MacOS
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Using Miniconda (alternative)
```bash
# Initialize conda for your shell (Only if you have not used Conda before)
conda init cmd.exe

# Navigate to your project folder
cd Backend

# Create the environment with Python 3.10
conda create -n lit-voice python=3.10 -y

# Activate the environment
conda activate lit-voice

# Install dependencies
conda install -c pytorch -c nvidia -c conda-forge fastapi uvicorn starlette httpx python-multipart python-dotenv pydantic-settings anyio numpy pandas librosa pysoundfile transformers pytorch torchvision torchaudio pytorch-cuda=12.1 redis-py pytest pytest-asyncio requests -y

# Start the backend server
uvicorn app.main:app --reload
```

### 5. Access the Application
Open your browser and navigate to [http://localhost:8080](http://localhost:8080)


## 📚 Project Structure

```
LIT-for-Voice/
├── Frontend/                # React frontend application
│   ├── components/          # React components
│   │   ├── analysis/        # Analysis and perturbation tools
│   │   ├── audio/           # Audio visualization components
│   │   ├── layout/          # Layout components
│   │   ├── panels/          # Dashboard panels
│   │   ├── ui/              # Reusable UI components
│   │   └── visualization/   # Data visualization components
│   ├── hooks/               # Custom React hooks
│   ├── lib/                 # Utility functions
│   └── pages/               # Page components
│
├── Backend/                 # FastAPI backend application
│   ├── app/                 # Application code
│   │   ├── api/             # API routes and endpoints
│   │   ├── core/            # Core functionality
│   │   └── services/        # Business logic services
│   ├── data/                # Sample datasets
│   ├── tests/               # Backend tests
│   └── uploads/             # User-uploaded audio files
│
├── CODE_OF_CONDUCT.md       # Community guidelines
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── README.md                # Project documentation
└── SECURITY.md              # Security policy
```

## 🧪 Available Scripts

### Frontend
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

### Backend
- `pytest` - Run backend tests
- `uvicorn app.main:app --reload` - Start the API server in development mode

## 🔍 Usage

1. **Upload Audio Data**: Use the audio uploader to load your audio files
2. **Select Models**: Choose from available audio models for analysis
3. **Explore Visualizations**:
   - Examine waveforms and spectrograms
   - View model predictions and confidence scores
   - Explore attention patterns and embedding spaces
   - Generate saliency maps to highlight important audio regions
4. **Apply Perturbations**: Test model robustness with various audio perturbations
5. **Analyze Results**: Use the interactive dashboard to gain insights

## 👥 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 🛡️ Security

For security-related issues, please refer to our [Security Policy](SECURITY.md).

## 👨‍💻 Authors

- **Anas Hussaindeen** - [GitHub Profile](https://github.com/AnasSAV)
- **Chandupa Ambepitiya** - [GitHub Profile](https://github.com/Chand2103)
- **Dewmike Amarasinghe** - [GitHub Profile](https://github.com/DewmikeAmarasinghe)

## 🙏 Acknowledgments

- Inspired by Google's [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)
- Built with modern React ecosystem and TypeScript
- Special thanks to the open-source community for the amazing tools and libraries

## 🛣️ Roadmap

- [ ] Backend API enhancements for model serving
- [ ] Support for more audio model architectures
- [ ] Advanced perturbation techniques
- [ ] Real-time audio processing capabilities
- [ ] Export functionality for visualizations
- [ ] Multi-language support
- [ ] Plugin system for custom analysis tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ for audio model interpretability</sub>
</p>
