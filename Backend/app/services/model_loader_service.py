import torch
from transformers import *
import librosa

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

    result = pipe(
        audio_file,
        generate_kwargs=generate_kwargs,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
    )
    return result

def transcribe_whisper_large():
    model_id = "openai/whisper-large-v3"
    return transcribe_whisper(model_id, "sample1.wav")

def transcribe_whisper_base():
    model_id = "openai/whisper-base"
    return transcribe_whisper(model_id, "sample1.wav")


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

def predict_emotion_wave2vec(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)  # Average over sequence length
        predicted_label = torch.argmax(predictions, dim=-1)
        emotion = model.config.id2label[predicted_label.item()]
    return emotion

def wave2vec():
    emotion = predict_emotion_wave2vec("sample2.mp3")
    return emotion



