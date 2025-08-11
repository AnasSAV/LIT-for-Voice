import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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

def transcribe_whisper_large(audio_file):
    model_id = "openai/whisper-large-v3"
    return transcribe_whisper(model_id, audio_file)

def transcribe_whisper_base(audio_file):
    model_id = "openai/whisper-base"
    return transcribe_whisper(model_id, audio_file)

