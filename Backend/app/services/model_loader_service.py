import logging
import torch
from transformers import (
    pipeline,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    WhisperProcessor,
    WhisperModel,
)
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)


def transcribe_whisper(model_id, audio_file, chunk_length_s=30, batch_size=8, return_timestamps=False):
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    audio, sample_rate = librosa.load(audio_file, sr=16000)
    audio = audio.astype(np.float32)

    result = pipe(
        audio,
        return_timestamps=return_timestamps,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
    )
    
    if return_timestamps:
        return {
            "text": result["text"],
            "chunks": result.get("chunks", []),
            "audio": audio,
            "sample_rate": sample_rate
        }
    return result["text"]

def transcribe_whisper_large(audio_file_path):
    model_id = "openai/whisper-large-v3"
    return transcribe_whisper(model_id, audio_file_path)

def transcribe_whisper_base(audio_file_path):
    model_id = "openai/whisper-base"
    return transcribe_whisper(model_id, audio_file_path)

def transcribe_whisper_with_timestamps(audio_file_path, model_size="base"):
    model_id = "openai/whisper-base" if model_size == "base" else "openai/whisper-large-v3"
    return transcribe_whisper(model_id, audio_file_path, return_timestamps=True)


_EMO_MODEL_ID = "r-f/wav2vec-english-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_EMO_MODEL_ID)
emo_model = Wav2Vec2ForSequenceClassification.from_pretrained(_EMO_MODEL_ID)
emo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
emo_model = emo_model.to(emo_device)

def predict_emotion_wave2vec(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

    # Move tensors to model device
    input_values = inputs.input_values.to(emo_device)
    attention_mask = inputs.attention_mask.to(emo_device) if "attention_mask" in inputs else None

    with torch.no_grad():
        outputs = emo_model(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, num_labels]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        label_idx = int(pred.item())
        
        # Get all emotion labels and their probabilities
        id2label = emo_model.config.id2label if isinstance(emo_model.config.id2label, dict) else {}
        emotion_probs = {}
        
        for i, prob in enumerate(probs[0]):
            emotion_label = id2label.get(i, f"emotion_{i}")
            emotion_probs[emotion_label] = float(prob.item())
        
        # Get the predicted emotion
        predicted_emotion = id2label.get(label_idx, str(label_idx))
        
        # Return both the prediction and all probabilities
        result = {
            "predicted_emotion": predicted_emotion,
            "probabilities": emotion_probs,
            "confidence": float(probs[0][label_idx].item())
        }
        
        logger.debug("Emotion logits shape=%s, predicted=%s, label=%s", tuple(logits.shape), label_idx, predicted_emotion)
    return result

def wave2vec(audio_file_path: str, return_probabilities: bool = False):
    """
    Predict emotion using wav2vec2 model.
    
    Args:
        audio_file_path: Path to the audio file
        return_probabilities: If True, returns detailed results with probabilities
    
    Returns:
        If return_probabilities=False: str (emotion label for backward compatibility)
        If return_probabilities=True: dict with prediction and probabilities
    """
    result = predict_emotion_wave2vec(audio_file_path)
    
    if return_probabilities:
        return result
    else:
        # Backward compatibility - return just the emotion string
        return result["predicted_emotion"]


# Whisper embeddings - Load models for embedding extraction
_whisper_processor_base = None
_whisper_model_base = None
_whisper_processor_large = None
_whisper_model_large = None

def get_whisper_base_models():
    global _whisper_processor_base, _whisper_model_base
    if _whisper_processor_base is None:
        _whisper_processor_base = WhisperProcessor.from_pretrained("openai/whisper-base")
        _whisper_model_base = WhisperModel.from_pretrained("openai/whisper-base")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _whisper_model_base = _whisper_model_base.to(device)
    return _whisper_processor_base, _whisper_model_base

def get_whisper_large_models():
    global _whisper_processor_large, _whisper_model_large
    if _whisper_processor_large is None:
        # Clear CUDA cache before loading large model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model with optimizations
        _whisper_processor_large = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        _whisper_model_large = WhisperModel.from_pretrained("openai/whisper-large-v3")
        _whisper_model_large = _whisper_model_large.to(device)
    return _whisper_processor_large, _whisper_model_large

def extract_whisper_embeddings(audio_file_path: str, model_size: str = "base") -> np.ndarray:
    """
    Extract Whisper encoder embeddings from audio file.
    Returns pooled encoder hidden states (mean pooling across time dimension).
    
    Args:
        audio_file_path: Path to audio file
        model_size: "base" or "large"
    
    Returns:
        numpy array of embeddings (512-dim for base, 1280-dim for large)
    """
    # Load audio
    audio, sample_rate = librosa.load(audio_file_path, sr=16000)
    audio = audio.astype(np.float32)
    
    if model_size == "base":
        processor, model = get_whisper_base_models()
    elif model_size == "large":
        processor, model = get_whisper_large_models()
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    device = next(model.parameters()).device
    
    # Process audio to log-mel spectrogram
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    with torch.no_grad():
        # Get encoder outputs
        encoder_outputs = model.encoder(input_features)
        # encoder_outputs.last_hidden_state shape: [batch, time_frames, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state
        
        # Mean pooling across time dimension to get single vector per clip
        pooled_embeddings = torch.mean(hidden_states, dim=1)  # [batch, hidden_size]
        
        # Convert to numpy
        embeddings = pooled_embeddings.cpu().numpy().squeeze()  # [hidden_size]
    
    return embeddings

def extract_wav2vec2_embeddings(audio_file_path: str) -> np.ndarray:
    """
    Extract Wav2Vec2 embeddings from audio file.
    Returns pooled hidden states from the last layer.
    
    Args:
        audio_file_path: Path to audio file
    
    Returns:
        numpy array of embeddings
    """
    # Load audio
    audio, rate = librosa.load(audio_file_path, sr=16000)
    
    # Use the same feature extractor as emotion model
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    # Move tensors to model device
    input_values = inputs.input_values.to(emo_device)
    attention_mask = inputs.attention_mask.to(emo_device) if "attention_mask" in inputs else None
    
    with torch.no_grad():
        # Get hidden states (before classification head)
        outputs = emo_model.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        # outputs.last_hidden_state shape: [batch, time_frames, hidden_size]
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling across time dimension
        pooled_embeddings = torch.mean(hidden_states, dim=1)  # [batch, hidden_size]
        
        # Convert to numpy
        embeddings = pooled_embeddings.cpu().numpy().squeeze()  # [hidden_size]
    
    return embeddings

def reduce_dimensions(embeddings_list: list, method: str = "pca", n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality of embeddings for visualization.
    
    Args:
        embeddings_list: List of embedding arrays
        method: "pca", "tsne", or "umap"
        n_components: Number of output dimensions (2 or 3)
    
    Returns:
        Reduced embeddings as numpy array [n_samples, n_components]
    """
    if not embeddings_list:
        return np.array([])
    
    # Stack embeddings into matrix
    X = np.vstack(embeddings_list)
    
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings_list)-1))
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=min(15, len(embeddings_list)-1))
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    
    reduced = reducer.fit_transform(X)
    return reduced