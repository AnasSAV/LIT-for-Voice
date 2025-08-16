import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification
)
import librosa
import numpy as np
import hashlib
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Global model cache to avoid reloading models
_model_cache: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading and caching for efficient inference"""
    
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
    
    def get_whisper_pipeline(self, model_id: str):
        """Get or create Whisper pipeline with caching"""
        cache_key = f"whisper_{model_id}"
        
        if cache_key not in _model_cache:
            logger.info(f"Loading Whisper model: {model_id}")
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            ).to(self.device)
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            _model_cache[cache_key] = pipe
            logger.info(f"Cached Whisper model: {model_id}")
        
        return _model_cache[cache_key]
    
    def get_wav2vec_models(self):
        """Get or create Wav2Vec2 models with caching"""
        cache_key = "wav2vec_emotion"
        
        if cache_key not in _model_cache:
            logger.info("Loading Wav2Vec2 emotion recognition model")
            
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "r-f/wav2vec-english-speech-emotion-recognition"
            )
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "r-f/wav2vec-english-speech-emotion-recognition"
            )
            
            _model_cache[cache_key] = {
                "feature_extractor": feature_extractor,
                "model": model
            }
            logger.info("Cached Wav2Vec2 emotion recognition model")
        
        return _model_cache[cache_key]

# Global model manager instance
model_manager = ModelManager()

def get_audio_hash(audio_path: str) -> str:
    """Generate hash for audio file for caching purposes"""
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    return hashlib.md5(audio_data).hexdigest()

def transcribe_whisper(model_id: str, audio_file: str, chunk_length_s: int = 30, batch_size: int = 8) -> Dict[str, Any]:
    """Transcribe audio using Whisper with caching"""
    
    # Get cached pipeline
    pipe = model_manager.get_whisper_pipeline(model_id)
    
    # Load audio using librosa instead of relying on ffmpeg
    audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
    
    generate_kwargs = {"return_timestamps": True}
    
    result = pipe(
        audio_array,
        generate_kwargs=generate_kwargs,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
    )
    
    return result

def predict_emotion_wave2vec(audio_path: str) -> str:
    """Predict emotion using Wav2Vec2 with caching"""
    
    # Get cached models
    models = model_manager.get_wav2vec_models()
    feature_extractor = models["feature_extractor"]
    model = models["model"]
    
    # Load and process audio
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1)
        
        # Define emotion labels manually since the model config might not have them
        emotion_labels = {
            0: "angry",
            1: "disgust", 
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }
        
        predicted_id = predicted_label.item()
        emotion = emotion_labels.get(predicted_id, f"unknown_emotion_{predicted_id}")
        
        logger.debug(f"Logits shape: {outputs.logits.shape}")
        logger.debug(f"Predicted emotion ID: {predicted_id}, emotion: {emotion}")
    
    return emotion

# Legacy functions for backward compatibility with sample files
def transcribe_whisper_large():
    """Transcribe using Whisper Large with sample file"""
    model_id = "openai/whisper-large-v3"
    return transcribe_whisper(model_id, "sample1.wav")

def transcribe_whisper_base():
    """Transcribe using Whisper Base with sample file"""
    model_id = "openai/whisper-base"
    return transcribe_whisper(model_id, "sample1.wav")

def wave2vec():
    """Emotion recognition with sample file"""
    return predict_emotion_wave2vec("sample2.mp3")

# New functions for file upload support
def transcribe_whisper_large_file(audio_file: str):
    """Transcribe uploaded file using Whisper Large"""
    model_id = "openai/whisper-large-v3"
    return transcribe_whisper(model_id, audio_file)

def transcribe_whisper_base_file(audio_file: str):
    """Transcribe uploaded file using Whisper Base"""
    model_id = "openai/whisper-base"
    return transcribe_whisper(model_id, audio_file)

def wave2vec_file(audio_file: str):
    """Emotion recognition for uploaded file"""
    return predict_emotion_wave2vec(audio_file)



