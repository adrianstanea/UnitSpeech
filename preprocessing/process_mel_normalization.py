import argparse
import gc
import logging
import os

import librosa
import torch
import torchaudio as ta
from tqdm import tqdm

from conf.hydra_config import (
    TrainingUnitEncoderConfig_STEP1,
)
from preprocessing.utils import load_and_process_wav
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN
from unitspeech.util import parse_filelist
from unitspeech.vocoder.meldataset import mel_spectrogram


def get_channel_wise_mel_normalization(filelist_path, mel_min, mel_max, cfg: TrainingUnitEncoderConfig_STEP1):
    filelist = parse_filelist(filelist_path, split_char='|')

    for idx, line in tqdm(enumerate(filelist, start=1), total=len(filelist)):
        filepath = line[0]
        
        wav, sr = librosa.load(filepath)
        wav = torch.FloatTensor(wav).unsqueeze(0)

        mel = mel_spectrogram(wav,
                                cfg.data.n_fft,
                                cfg.data.n_feats,
                                cfg.data.sampling_rate,
                                cfg.data.hop_length,
                                cfg.data.win_length,
                                cfg.data.mel_fmin,
                                cfg.data.mel_fmax,
                                center=False)
        channels_min = mel.min(-1, keepdim=False)[0]
        channels_max = mel.max(-1, keepdim=False)[0]

        mel_min = torch.min(mel_min, channels_min)
        mel_max = torch.max(mel_max, channels_max)
    return mel_min, mel_max

def main():
    cfg = TrainingUnitEncoderConfig_STEP1
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    folder_path = os.path.dirname(cfg.dataset.mel_max_path)
    if not os.path.exists(folder_path):
        print(f"Created directory {folder_path}")
        os.makedirs(folder_path)
    else:
        print(f"Directory {folder_path} already exists")

    mel_min = torch.full((cfg.data.n_feats,), float('inf'))
    mel_max = torch.full((cfg.data.n_feats,), float('-inf'))

    print(f"Loading filelist from {cfg.dataset.train_filelist_path}")
    mel_min, mel_max = get_channel_wise_mel_normalization(cfg.dataset.train_filelist_path,
                                                          mel_min,
                                                          mel_max,
                                                          cfg)

    print(f"Loading filelist from {cfg.dataset.test_filelist_path}")
    mel_min, mel_max = get_channel_wise_mel_normalization(cfg.dataset.test_filelist_path,
                                                          mel_min,
                                                          mel_max,
                                                          cfg)

    # Log and save run data
    mel_min = mel_min.squeeze(0)
    mel_max = mel_max.squeeze(0)
    print(f"Mel min of dataset: {mel_min}")
    print(f"Mel max of dataset: {mel_max}")

    print(f"Saving mel min to {cfg.dataset.mel_min_path}")
    torch.save(mel_min, cfg.dataset.mel_min_path)

    print(f"Saving mel max to {cfg.dataset.mel_max_path}")
    torch.save(mel_max, cfg.dataset.mel_max_path)

if __name__ == '__main__':
    main()