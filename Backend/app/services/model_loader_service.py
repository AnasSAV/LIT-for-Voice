import torch
from transformers import *
import librosa
import numpy as np
import threading
from collections import defaultdict
import gc

# Global device/dtype for this process
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Lazy caches for ASR (Whisper) pipelines
_asr_pipelines = {}
_asr_locks = defaultdict(threading.Lock)

def get_asr_pipeline(model_id: str):
    pipe = _asr_pipelines.get(model_id)
    if pipe is not None:
        return pipe
    lock = _asr_locks[model_id]
    with lock:
        pipe = _asr_pipelines.get(model_id)
        if pipe is not None:
            return pipe
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=_torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(_device)

        processor = AutoProcessor.from_pretrained(model_id)

        p = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=_torch_dtype,
            device=_device,
        )
        _asr_pipelines[model_id] = p
        return p

def transcribe_whisper(model_id, audio_file, chunk_length_s=30, batch_size=8):
    pipe = get_asr_pipeline(model_id)

    generate_kwargs = {"return_timestamps": True}

    audio, sample_rate = librosa.load(audio_file, sr=16000)
    audio = audio.astype(np.float32)

    result = pipe(
        audio,
        generate_kwargs=generate_kwargs,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
    )
    
    return result["text"]

def transcribe_whisper_large(audio_file_path):
    model_id = "openai/whisper-large-v3"
    return transcribe_whisper(model_id, audio_file_path)

def transcribe_whisper_base(audio_file_path):
    model_id = "openai/whisper-base"
    return transcribe_whisper(model_id, audio_file_path)


# Lazy cache for wav2vec2 emotion model
_emotion_feature_extractor = None
_emotion_model = None
_emotion_lock = threading.Lock()

def get_emotion_components():
    global _emotion_feature_extractor, _emotion_model
    if _emotion_feature_extractor is not None and _emotion_model is not None:
        return _emotion_feature_extractor, _emotion_model
    with _emotion_lock:
        if _emotion_feature_extractor is not None and _emotion_model is not None:
            return _emotion_feature_extractor, _emotion_model
        model_id = "r-f/wav2vec-english-speech-emotion-recognition"
        _emotion_feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        _emotion_model = AutoModelForAudioClassification.from_pretrained(model_id).to(_device)
        _emotion_model.eval()
        return _emotion_feature_extractor, _emotion_model

def predict_emotion_wave2vec(audio_path):
    feature_extractor, emotion_model = get_emotion_components()
    # Load mono audio at model's expected sampling rate
    audio, rate = librosa.load(audio_path, sr=16000, mono=True)
    # Feature extraction without padding for single clip; include attention mask
    inputs = feature_extractor(
        audio,
        sampling_rate=rate,
        return_tensors="pt",
        padding=False,
        return_attention_mask=True,
    )
    # Move inputs to model device
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        # Pass all inputs so attention_mask is respected
        outputs = emotion_model(**inputs)
        logits = outputs.logits[0]  # shape: (num_labels,)
        probs_tensor = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label_idx = int(torch.argmax(probs_tensor).item())
        id2label = emotion_model.config.id2label
        emotion = id2label[predicted_label_idx]
        # Build per-class probability mapping
        probs = {id2label[i]: float(probs_tensor[i].item()) for i in range(probs_tensor.shape[0])}
        confidence = float(probs[emotion])
    return {
        "label": emotion,
        "confidence": confidence,
        "probs": probs,
        "model_id": "r-f/wav2vec-english-speech-emotion-recognition",
        "params": {},
    }

def wave2vec(audio_file_path):
    # Returns a structured dict with label/confidence/probs
    return predict_emotion_wave2vec(audio_file_path)


# ------------------------------
# Unload helpers (manual flush)
# ------------------------------

def unload_asr_model(model_id: str | None = None) -> int:
    """
    Unload a specific Whisper pipeline by HF model id (e.g., "openai/whisper-base"),
    or unload all ASR pipelines if model_id is None. Returns the number of pipelines
    removed from cache. If running on CUDA, empties the CUDA cache.
    """
    removed = 0
    global _asr_pipelines
    if model_id is None:
        # Unload all
        ids = list(_asr_pipelines.keys())
        for mid in ids:
            lock = _asr_locks[mid]
            with lock:
                pipe = _asr_pipelines.pop(mid, None)
                if pipe is not None:
                    try:
                        # Move underlying model to CPU to release VRAM faster
                        if hasattr(pipe, "model") and hasattr(pipe.model, "to"):
                            pipe.model.to("cpu")
                    except Exception:
                        pass
                    del pipe
                    removed += 1
    else:
        lock = _asr_locks[model_id]
        with lock:
            pipe = _asr_pipelines.pop(model_id, None)
            if pipe is not None:
                try:
                    if hasattr(pipe, "model") and hasattr(pipe.model, "to"):
                        pipe.model.to("cpu")
                except Exception:
                    pass
                del pipe
                removed = 1

    # Encourage memory release
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    return removed


def unload_emotion_model() -> bool:
    """
    Unload the wav2vec2 emotion model/components from cache.
    Returns True if anything was removed.
    """
    global _emotion_feature_extractor, _emotion_model
    removed = False
    with _emotion_lock:
        if _emotion_model is not None:
            try:
                _emotion_model.to("cpu")
            except Exception:
                pass
            _emotion_model = None
            removed = True
        if _emotion_feature_extractor is not None:
            _emotion_feature_extractor = None
            removed = True

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()
    return removed


def unload_all_models() -> dict:
    """
    Unload all cached models (ASR pipelines and emotion model). Returns a summary dict.
    """
    asr_removed = unload_asr_model(None)
    emo_removed = unload_emotion_model()
    return {"asr_removed": asr_removed, "emotion_removed": emo_removed}