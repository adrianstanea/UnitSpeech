import os

import librosa
import torch
import torchaudio


def load_and_process_wav(filepath, device):
    wav, sr = librosa.load(filepath)
    wav = torch.FloatTensor(wav).to(device)
    if sr != 16_000:
        resample_fn = torchaudio.transforms.Resample(sr, 16_000).cuda()
        wav = resample_fn(wav)
    return wav

def save_mean_emb(crnt_spkr_mean, crnt_speaker, all_mean_embs, dataset_name):
    all_mean_embs[int(crnt_speaker)] = crnt_spkr_mean.unsqueeze(1)
    os.makedirs(f"resources/{dataset_name}/speaker_embs", exist_ok=True)
    torch.save(crnt_spkr_mean, f"resources/{dataset_name}/speaker_embs/{crnt_speaker}.pt")