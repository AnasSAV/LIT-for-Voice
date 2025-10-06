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

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
        )
    except NotImplementedError as e:
        if "meta tensor" in str(e):
            # Fallback for meta tensor issue: load on CPU first then move to CUDA
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=-1,  # Force CPU first
            )
            if torch.cuda.is_available():
                try:
                    pipe.model = pipe.model.to("cuda:0")
                except Exception:
                    pass  # Stay on CPU if move fails
        else:
            raise
    audio, sample_rate = librosa.load(audio_file, sr=16000)
    audio = audio.astype(np.float32)

    if return_timestamps:
        
        result = pipe(
            audio,
            return_timestamps="word",  # Get word-level timestamps instead of chunk-level
            chunk_length_s=5,  # Use smaller chunks (5 seconds instead of 30)
            batch_size=batch_size,
        )
    else:
        # For regular transcription, use original parameters
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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _whisper_processor_base = WhisperProcessor.from_pretrained("openai/whisper-base")
        _whisper_model_base = WhisperModel.from_pretrained("openai/whisper-base")
        _whisper_model_base = _whisper_model_base.to(device)
    return _whisper_processor_base, _whisper_model_base

def get_whisper_large_models():
    global _whisper_processor_large, _whisper_model_large
    if _whisper_processor_large is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _whisper_processor_large = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        try:
            _whisper_model_large = WhisperModel.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            _whisper_model_large = _whisper_model_large.to(device)
        except NotImplementedError as e:
            if "meta tensor" in str(e):
                # Handle meta tensor issue for embeddings model too
                _whisper_model_large = WhisperModel.from_pretrained("openai/whisper-large-v3")
                _whisper_model_large = _whisper_model_large.to(device)
            else:
                raise
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


def extract_audio_frequency_features(audio_file_path: str) -> dict:
    """
    Extract comprehensive frequency-domain audio features using librosa.
    
    Args:
        audio_file_path: Path to audio file
    
    Returns:
        Dictionary containing various audio frequency features
    """
    # Load audio with standard sample rate
    audio, sr = librosa.load(audio_file_path, sr=22050)
    
    # Extract various frequency features
    features = {}
    
    # Basic spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    
    # MFCC features (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Chroma features (pitch class profiles)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Tonnetz (tonal centroid features)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    
    # Tempo and beat tracking
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    
    # Calculate statistics for each feature
    features = {
        "spectral_centroid": {
            "mean": float(np.mean(spectral_centroids)),
            "std": float(np.std(spectral_centroids)),
            "min": float(np.min(spectral_centroids)),
            "max": float(np.max(spectral_centroids))
        },
        "spectral_rolloff": {
            "mean": float(np.mean(spectral_rolloff)),
            "std": float(np.std(spectral_rolloff)),
            "min": float(np.min(spectral_rolloff)),
            "max": float(np.max(spectral_rolloff))
        },
        "spectral_bandwidth": {
            "mean": float(np.mean(spectral_bandwidth)),
            "std": float(np.std(spectral_bandwidth)),
            "min": float(np.min(spectral_bandwidth)),
            "max": float(np.max(spectral_bandwidth))
        },
        "zero_crossing_rate": {
            "mean": float(np.mean(zero_crossing_rate)),
            "std": float(np.std(zero_crossing_rate)),
            "min": float(np.min(zero_crossing_rate)),
            "max": float(np.max(zero_crossing_rate))
        },
        "rms_energy": {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "min": float(np.min(rms)),
            "max": float(np.max(rms))
        },
        "mfcc": {
            f"mfcc_{i+1}_mean": float(np.mean(mfccs[i])) for i in range(13)
        },
        "chroma": {
            f"chroma_{i+1}_mean": float(np.mean(chroma[i])) for i in range(12)
        },
        "tonnetz": {
            f"tonnetz_{i+1}_mean": float(np.mean(tonnetz[i])) for i in range(6)
        },
        "tempo": float(tempo),
        "duration": float(len(audio) / sr),
        "sample_rate": int(sr)
    }
    
    # Flatten the nested structure for easier processing
    flattened_features = {}
    for key, value in features.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened_features[f"{key}_{subkey}"] = subvalue
        else:
            flattened_features[key] = value
    
    return flattened_features


def extract_whisper_attention_pairs(audio_file_path: str, model_size: str = "base", layer_idx: int = 6, head_idx: int = 0) -> dict:
    """
    Extract word-to-word attention relationships from Whisper model.
    
    Args:
        audio_file_path: Path to audio file
        model_size: "base" or "large"
        layer_idx: Which transformer layer to extract attention from (0-11 for base)
        head_idx: Which attention head to use (0-11 for base)
    
    Returns:
        Dictionary with attention pairs and timestamp-level attention
    """
    logger.info(f"Extracting attention pairs from Whisper {model_size} for {audio_file_path}")
    
    # Get word timestamps first
    try:
        timestamp_data = transcribe_whisper_with_timestamps(audio_file_path, model_size)
        chunks = timestamp_data.get("chunks", [])
        audio = timestamp_data.get("audio")
        total_duration = len(audio) / 16000.0 if audio is not None else 0.0
        
        logger.info(f"Got {len(chunks)} word chunks, total duration: {total_duration:.2f}s")
    except Exception as e:
        logger.error(f"Failed to get timestamps: {e}")
        return {"error": "Failed to extract timestamps", "attention_pairs": [], "timestamp_attention": []}
    
    if not chunks:
        logger.warning("No word chunks found")
        return {"attention_pairs": [], "timestamp_attention": [], "total_duration": 0}
    
    try:
        # Get Whisper models
        if model_size == "base":
            processor, model = get_whisper_base_models()
        else:
            processor, model = get_whisper_large_models()
        
        device = next(model.parameters()).device
        
        # Process audio to get input features
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        # Forward pass with attention extraction
        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_features,
                output_attentions=True  # Extract attention weights
            )
            
            # Get attention weights: tuple of tensors [num_layers]
            # Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
            attention_weights = encoder_outputs.attentions
            
        # Validate layer and head indices
        num_layers = len(attention_weights)
        if layer_idx >= num_layers:
            layer_idx = num_layers - 1
            logger.warning(f"Layer index too high, using layer {layer_idx}")
            
        # Get specific layer attention: [batch_size, num_heads, seq_len, seq_len]
        layer_attention = attention_weights[layer_idx]
        num_heads = layer_attention.shape[1]
        
        if head_idx >= num_heads:
            head_idx = 0
            logger.warning(f"Head index too high, using head {head_idx}")
        
        # Extract attention matrix: [seq_len, seq_len]
        attention_matrix = layer_attention[0, head_idx].cpu().numpy()
        seq_len = attention_matrix.shape[0]
        
        logger.info(f"Extracted attention matrix: {attention_matrix.shape} from layer {layer_idx}, head {head_idx}")
        
        # Map time frames to words
        fps = seq_len / total_duration if total_duration > 0 else 1.0
        
        # Generate word-to-word attention pairs
        word_pairs = []
        for i, word1 in enumerate(chunks):
            for j, word2 in enumerate(chunks):
                try:
                    # Get time boundaries
                    start1, end1 = word1.get("timestamp", [0, 0])
                    start2, end2 = word2.get("timestamp", [0, 0])
                    
                    # Convert to frame indices
                    frame1_start = max(0, min(seq_len-1, int(start1 * fps)))
                    frame1_end = max(frame1_start+1, min(seq_len, int(end1 * fps)))
                    frame2_start = max(0, min(seq_len-1, int(start2 * fps)))
                    frame2_end = max(frame2_start+1, min(seq_len, int(end2 * fps)))
                    
                    # Extract attention weight between word regions
                    attention_region = attention_matrix[frame1_start:frame1_end, frame2_start:frame2_end]
                    attention_score = float(np.mean(attention_region)) if attention_region.size > 0 else 0.0
                    
                    word_pairs.append({
                        "from_word": word1.get("text", "").strip(),
                        "to_word": word2.get("text", "").strip(),
                        "from_time": [float(start1), float(end1)],
                        "to_time": [float(start2), float(end2)],
                        "attention_weight": attention_score,
                        "from_index": i,
                        "to_index": j
                    })
                except Exception as e:
                    logger.error(f"Error processing word pair {i}-{j}: {e}")
                    continue
        
        # Generate timestamp-level attention (continuous timeline)
        timestamp_attention = []
        time_resolution = 0.1  # 100ms resolution
        for t in np.arange(0, total_duration, time_resolution):
            frame_idx = max(0, min(seq_len-1, int(t * fps)))
            
            # Average attention from this frame to all other frames
            avg_attention = float(np.mean(attention_matrix[frame_idx, :]))
            
            timestamp_attention.append({
                "time": float(t),
                "attention": avg_attention,
                "frame_index": frame_idx
            })
        
        logger.info(f"Generated {len(word_pairs)} attention pairs and {len(timestamp_attention)} timestamp points")
        
        return {
            "model": f"whisper-{model_size}",
            "layer": layer_idx,
            "head": head_idx,
            "attention_pairs": word_pairs,
            "timestamp_attention": timestamp_attention,
            "total_duration": float(total_duration),
            "sequence_length": seq_len,
            "word_chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"Error extracting attention pairs: {e}")
        return {
            "error": f"Failed to extract attention: {str(e)}",
            "attention_pairs": [], 
            "timestamp_attention": [],
            "total_duration": float(total_duration) if 'total_duration' in locals() else 0
        }

