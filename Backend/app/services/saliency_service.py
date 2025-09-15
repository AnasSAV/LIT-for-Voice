import logging
import os
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from captum.attr import IntegratedGradients, GradientShap, Lime
from captum.attr._utils.lrp_rules import EpsilonRule
from captum.attr._core.lrp import LRP
from app.services.model_loader_service import (
    transcribe_whisper_base,
    transcribe_whisper_large,
    transcribe_whisper_with_timestamps,
    predict_emotion_wave2vec,
    get_whisper_base_models,
    get_whisper_large_models,
    feature_extractor,
    emo_model,
    emo_device
)

logger = logging.getLogger(__name__)
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))  # cap analysis window
MAX_SALIENCY_SECONDS_SHAP = int(os.getenv("MAX_SALIENCY_SECONDS_SHAP", "6"))  # stricter for SHAP
SALIENCY_SHAP_SAMPLES = int(os.getenv("SALIENCY_SHAP_SAMPLES", "8"))

def detect_model_type(model: str) -> str:
    if "whisper" in model.lower():
        return "whisper"
    elif "wav2vec" in model.lower():
        return "wav2vec2"
    return "unknown"


def generate_whisper_saliency(audio_file_path: str, model_size: str = "base", method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    if existing_prediction and "chunks" in existing_prediction:
        data = existing_prediction
        audio = data["audio"]
        chunks = data["chunks"]
    else:
        data = transcribe_whisper_with_timestamps(audio_file_path, model_size)
        audio = data["audio"]
        chunks = data["chunks"]
    
    # Crop to a safe max duration to avoid OOM
    if isinstance(audio, (list, tuple)):
        audio = np.asarray(audio)
    if hasattr(audio, "shape") and audio is not None:
        max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
        max_len = int(max_seconds * 16000)
        if len(audio) > max_len:
            audio = audio[:max_len]
            # Keep only chunks inside the window
            chunks = [c for c in chunks if c.get("timestamp", [0, 0])[0] < max_seconds]
    
    if model_size == "base":
        processor, model = get_whisper_base_models()
    else:
        processor, model = get_whisper_large_models()
    
    device = next(model.parameters()).device
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    input_features.requires_grad_(True)
    
    def model_forward(inputs):
        # Reduce to a scalar per batch: energy of encoder activations
        enc = model.encoder(inputs).last_hidden_state  # [B, T, H]
        return enc.pow(2).mean(dim=(1, 2))             # [B]
    
    if method == "gradcam":
        # Optimize memory usage for GPU
        torch.cuda.empty_cache()
        
        # Use gradient checkpointing to save memory
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        
        # Use smaller batch size and fewer steps to fit in GPU memory
        n_steps = 16  # Reduced from 32 to 16
        internal_batch_size = 1
        
        # Monitor GPU memory
        if torch.cuda.is_available():
            logger.info(f"GPU memory before saliency: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        try:
            ig = IntegratedGradients(model_forward)
            attributions = ig.attribute(
                input_features,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                # Clear cache and try again with even lower memory settings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("First attempt failed, trying with even lower memory settings...")
                
                # Reduce memory usage further
                n_steps = 8
                
                # Try with gradient accumulation
                try:
                    ig = IntegratedGradients(model_forward)
                    attributions = ig.attribute(
                        input_features,
                        n_steps=n_steps,
                        internal_batch_size=internal_batch_size,
                    )
                except RuntimeError as e2:
                    logger.error(f"Saliency computation failed on GPU: {str(e2)}")
                    raise RuntimeError("Failed to compute saliency on GPU after optimization attempts") from e2
            else:
                raise
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_features)
    elif method == "shap":
        # Use Captum GradientShap on the model's current device with small n_samples
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_features)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attributions = gs.attribute(
                input_features,
                baselines=baseline,
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("Whisper SHAP OOM; retrying with fewer samples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                attributions = gs.attribute(
                    input_features,
                    baselines=baseline,
                    n_samples=max(2, min(8, SALIENCY_SHAP_SAMPLES // 2 if SALIENCY_SHAP_SAMPLES > 2 else 2)),
                    stdevs=0.07,
                )
            else:
                logger.exception("Whisper SHAP failed; falling back to energy map")
                attributions = None
        except Exception:
            logger.exception("Whisper SHAP failed; falling back to energy map")
            attributions = None
    else:
        attributions = torch.zeros_like(input_features)
    
    # Reduce to 1D timeline and normalize to [0,1] for visible intensities
    if attributions is not None:
        saliency_np = attributions.detach().cpu().numpy().squeeze()
        if saliency_np.ndim == 2:
            if saliency_np.shape[0] in (64, 80, 128):
                agg = np.mean(np.abs(saliency_np), axis=0)
            else:
                agg = np.mean(np.abs(saliency_np), axis=1)
        elif saliency_np.ndim == 1:
            agg = np.abs(saliency_np)
        else:
            while saliency_np.ndim > 1:
                saliency_np = saliency_np.mean(axis=0)
            agg = np.abs(saliency_np)
        max_abs = float(np.max(agg)) if agg.size > 0 else 0.0
        saliency_scores = (agg / max_abs) if max_abs > 0 else np.zeros_like(agg)
    else:
        saliency_scores = np.array([])

    # Fallback: if scores are empty or nearly constant, use encoder energy map
    use_energy_fallback = (
        saliency_scores.size == 0 or
        (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6
    )
    if use_energy_fallback:
        logger.info("Using Whisper energy-map fallback for saliency")
        with torch.no_grad():
            enc = model.encoder(input_features).last_hidden_state  # [B, T, H]
            energy = enc.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)

    # Create dense series with smoothing and percentile clipping
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
    
    segments = []
    # Map timestamps to attribution timeline robustly
    total_duration = float(len(audio)) / 16000.0 if hasattr(audio, "__len__") and len(audio) > 0 else 0.0
    T = len(saliency_scores)
    fps = (T / total_duration) if total_duration > 0 else 1.0
    
    for chunk in chunks or []:
        start_time = chunk.get("timestamp", [0, 0])[0]
        end_time = chunk.get("timestamp", [0, 0])[1]
        word = chunk.get("text", "")
        
        # Convert to attribution frames using derived fps
        start_frame = max(0, min(T - 1, int(start_time * fps)))
        end_frame = max(0, min(T, int(end_time * fps)))
        
        if end_frame > start_frame:
            segment_saliency = float(np.mean(saliency_scores[start_frame:end_frame]))
            segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "word": word,
                "saliency": segment_saliency,
                "intensity": float(abs(segment_saliency))
            })

    # Fallback: if no text chunks or mapping produced 0 segments, create uniform segments
    if len(segments) == 0 and T > 0 and total_duration > 0:
        num_segments = max(8, min(64, int(total_duration * 4)))
        for i in range(num_segments):
            s_time = (i / num_segments) * total_duration
            e_time = ((i + 1) / num_segments) * total_duration
            s_idx = int(s_time * fps)
            e_idx = int(e_time * fps)
            s_idx = max(0, min(T - 1, s_idx))
            e_idx = max(s_idx + 1, min(T, e_idx))
            seg_sal = float(np.mean(saliency_scores[s_idx:e_idx]))
            segments.append({
                "start_time": s_time,
                "end_time": e_time,
                "saliency": seg_sal,
                "intensity": float(abs(seg_sal)),
            })

    # Final normalization across segments for visibility
    if len(segments) > 0:
        vals = np.array([s.get("saliency", 0.0) for s in segments], dtype=np.float32)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        vrng = vmax - vmin
        if vrng < 1e-9:
            # Use absolute values spread if flat
            vals = np.abs(vals)
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            vrng = vmax - vmin
        for s, v in zip(segments, vals):
            s["intensity"] = float((v - vmin) / (vrng + 1e-9))
    
    return {
        "model": f"whisper-{model_size}",
        "method": method,
        "segments": segments,
        "total_duration": total_duration,
        "series": series.tolist()
    }

def generate_wav2vec2_saliency(audio_file_path: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    audio, rate = librosa.load(audio_file_path, sr=16000)
    # Crop to safe max duration to bound memory
    max_seconds = MAX_SALIENCY_SECONDS_SHAP if method == "shap" else MAX_SALIENCY_SECONDS
    max_len = int(max_seconds * rate)
    if len(audio) > max_len:
        audio = audio[:max_len]
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    input_values = inputs.input_values.to(emo_device)
    attention_mask = inputs.attention_mask.to(emo_device) if "attention_mask" in inputs else None
    
    input_values.requires_grad_(True)
    
    # Determine class to attribute (predicted emotion)
    with torch.no_grad():
        tmp_out = emo_model(input_values=input_values, attention_mask=attention_mask)
        tmp_probs = torch.nn.functional.softmax(tmp_out.logits, dim=-1)
        target_idx = int(torch.argmax(tmp_probs, dim=-1).item())

    def model_forward(inputs, mask=None, cls_idx: int = 0):
        outputs = emo_model(input_values=inputs, attention_mask=mask)
        return outputs.logits[:, cls_idx]
    
    if method == "gradcam":
        ig = IntegratedGradients(model_forward)
        try:
            attributions = ig.attribute(
                input_values,
                additional_forward_args=(attention_mask, target_idx),
                n_steps=32,
                internal_batch_size=1,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA OOM during Wav2Vec2 saliency. Falling back to CPU with fewer steps.")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu_device = torch.device("cpu")
                    # Move model and inputs to CPU
                    if hasattr(emo_model, 'to'):
                        emo_model.to(cpu_device)
                    input_values_cpu = input_values.detach().to(cpu_device)
                    input_values_cpu.requires_grad_(True)
                    attention_mask_cpu = attention_mask.detach().to(cpu_device) if attention_mask is not None else None
                    attributions = ig.attribute(
                        input_values_cpu,
                        additional_forward_args=(attention_mask_cpu, target_idx),
                        n_steps=16,
                        internal_batch_size=1,
                    )
                except Exception:
                    raise
            else:
                raise
    elif method == "lime":
        lime = Lime(model_forward)
        attributions = lime.attribute(input_values, additional_forward_args=(attention_mask, target_idx))
    elif method == "shap":
        # Use Captum GradientShap on model's current device with small n_samples
        gs = GradientShap(model_forward)
        baseline = torch.zeros_like(input_values)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attributions = gs.attribute(
                input_values,
                baselines=baseline,
                additional_forward_args=(attention_mask, target_idx),
                n_samples=max(2, min(16, SALIENCY_SHAP_SAMPLES)),
                stdevs=0.09,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning("Wav2Vec2 SHAP OOM; retrying with fewer samples")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                attributions = gs.attribute(
                    input_values,
                    baselines=baseline,
                    additional_forward_args=(attention_mask, target_idx),
                    n_samples=max(2, min(8, SALIENCY_SHAP_SAMPLES // 2 if SALIENCY_SHAP_SAMPLES > 2 else 2)),
                    stdevs=0.07,
                )
            else:
                logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
                attributions = None
        except Exception:
            logger.exception("Wav2Vec2 SHAP failed; falling back to energy map")
            attributions = None
    else:
        attributions = torch.zeros_like(input_values)
    
    # Normalize to [0,1] for visible intensities
    if attributions is not None:
        tmp = attributions.detach().cpu().numpy().squeeze()
        if tmp.ndim > 1:
            tmp = np.mean(np.abs(tmp), axis=0)
        else:
            tmp = np.abs(tmp)
        mx = float(np.max(tmp)) if tmp.size > 0 else 0.0
        saliency_scores = (tmp / mx) if mx > 0 else np.zeros_like(tmp)
    else:
        saliency_scores = np.array([])

    # Fallback: if SHAP produced empty/flat attributions, use encoder energy
    if saliency_scores.size == 0 or (np.max(saliency_scores) - np.min(saliency_scores) if saliency_scores.size > 0 else 0.0) < 1e-6:
        logger.info("Using Wav2Vec2 energy-map fallback for saliency")
        with torch.no_grad():
            hs = emo_model.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state  # [B,T,H]
            energy = hs.abs().mean(dim=2).squeeze(0).detach().cpu().numpy()
        if energy.size > 0:
            e_min, e_ptp = float(np.min(energy)), float(np.ptp(energy))
            saliency_scores = (energy - e_min) / (e_ptp + 1e-9)
        else:
            saliency_scores = np.zeros(1, dtype=np.float32)

    # Create dense series with smoothing and percentile clipping
    series = saliency_scores.astype(np.float32)
    if series.size > 0:
        win = max(3, int(series.size / 64))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float32) / float(win)
        series = np.convolve(series, kernel, mode="same")
        p95 = float(np.percentile(series, 95))
        if p95 > 0:
            series = np.clip(series, 0, p95)
        smin, smax = float(np.min(series)), float(np.max(series))
        series = (series - smin) / (smax - smin + 1e-9)
    
    with torch.no_grad():
        model_device = next(emo_model.parameters()).device
        iv = input_values.to(model_device)
        am = attention_mask.to(model_device) if attention_mask is not None else None
        outputs = emo_model(input_values=iv, attention_mask=am)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_emotion = torch.argmax(probs, dim=-1).item()
        id2label = emo_model.config.id2label
        emotion = id2label.get(predicted_emotion, str(predicted_emotion))
    
    segment_duration = len(audio) / 16000
    num_segments = 32
    segment_length = segment_duration / num_segments if num_segments > 0 else segment_duration
    
    segments = []
    # Map segment times to attribution indices using derived fps from saliency timeline
    T = len(saliency_scores)
    fps = (T / segment_duration) if segment_duration > 0 else 1.0
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        
        start_frame = max(0, min(T - 1, int(start_time * fps)))
        end_frame = max(start_frame + 1, min(T, int(end_time * fps)))
        segment_saliency = np.mean(saliency_scores[start_frame:end_frame])
        segments.append({
            "start_time": start_time,
            "end_time": end_time,
            "saliency": float(segment_saliency),
            "intensity": float(abs(segment_saliency))
        })

    # Final normalization across segments for visibility
    if len(segments) > 0:
        vals = np.array([s.get("saliency", 0.0) for s in segments], dtype=np.float32)
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        vrng = vmax - vmin
        if vrng < 1e-9:
            vals = np.abs(vals)
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            vrng = vmax - vmin
        for s, v in zip(segments, vals):
            s["intensity"] = float((v - vmin) / (vrng + 1e-9))
    
    return {
        "model": "wav2vec2",
        "method": method,
        "emotion": emotion,
        "segments": segments,
        "total_duration": segment_duration,
        "series": series.tolist()
    }

def generate_saliency(audio_file_path: str, model: str, method: str = "gradcam", existing_prediction: Dict = None) -> Dict:
    model_type = detect_model_type(model)
    
    if model_type == "whisper":
        model_size = "base" if "base" in model else "large"
        return generate_whisper_saliency(audio_file_path, model_size, method, existing_prediction)
    elif model_type == "wav2vec2":
        return generate_wav2vec2_saliency(audio_file_path, method, existing_prediction)
    else:
        raise ValueError(f"Unsupported model: {model}")
