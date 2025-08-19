import torch
from transformers import *
import librosa
import numpy as np

def transcribe_whisper(model_id, audio_file, chunk_length_s=30, batch_size=8):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
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


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

def predict_emotion_wave2vec(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)  # Average over sequence length
        predicted_label_idx = torch.argmax(predictions, dim=-1).item()
        emotion = model.config.id2label[predicted_label_idx]
        # Build per-class probability mapping
        probs_tensor = predictions[0]
        id2label = model.config.id2label
        probs = {id2label[i]: float(probs_tensor[i].item()) for i in range(probs_tensor.shape[0])}
        confidence = float(probs[id2label[predicted_label_idx]])
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



