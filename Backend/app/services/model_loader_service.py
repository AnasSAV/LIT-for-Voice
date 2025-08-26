import logging
import torch
from transformers import (
    pipeline,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)
import librosa
import numpy as np

logger = logging.getLogger(__name__)


def transcribe_whisper(model_id, audio_file, chunk_length_s=30, batch_size=8):
    # Use integer device indices to avoid meta-tensor issues in some transformer/torch combos
    device = 0 if torch.cuda.is_available() else -1  # 0 => cuda:0, -1 => CPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )

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
        # Map to label; fall back safely if out-of-range
        emotion = emo_model.config.id2label.get(label_idx, str(label_idx)) if isinstance(emo_model.config.id2label, dict) else str(label_idx)
        logger.debug("Emotion logits shape=%s, predicted=%s, label=%s", tuple(logits.shape), label_idx, emotion)
    return emotion

def wave2vec(audio_file_path: str) -> str:
    return predict_emotion_wave2vec(audio_file_path)



