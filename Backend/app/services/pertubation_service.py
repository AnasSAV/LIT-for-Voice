import torch
import torchaudio

def add_gaussian_noise(waveform, noise_level=0.005):
    """
    waveform: Tensor [channels, time]
    noise_level: Standard deviation of noise
    """
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def perturb(file_path, noise_level=0.005):
    waveform, sample_rate = torchaudio.load(file_path)  


    noisy_waveform = add_gaussian_noise(waveform, noise_level)
    
    return noisy_waveform