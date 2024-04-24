import argparse
import gc
import logging
import os

import numpy as np
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


def validate_channel_wise_mel_normalization(filelist_path, global_mel_min, global_mel_max, cfg: TrainingUnitEncoderConfig_STEP1):
    filelist = parse_filelist(filelist_path, split_char='|')

    for idx, line in tqdm(enumerate(filelist, start=1), total=len(filelist)):
        filepath = line[0]

        audio, sr = ta.load(filepath)

        if sr != cfg.data.sampling_rate:
            resample_fn = ta.transforms.Resample(sr, cfg.data.sampling_rate)
            audio = resample_fn(audio)
        mel = mel_spectrogram(audio,
                                cfg.data.n_fft,
                                cfg.data.n_feats,
                                cfg.data.sampling_rate,
                                cfg.data.hop_length,
                                cfg.data.win_length,
                                cfg.data.mel_fmin,
                                cfg.data.mel_fmax,
                                center=False).squeeze()
        mel = (mel - global_mel_min) / (global_mel_max - global_mel_min) * 2 - 1

        if np.all(np.logical_and(mel.numpy() >= -1, mel.numpy() <= 1)):
            continue
        else:
            print("ERROR: normalization failed for file", filepath)
    print("Normalization validation passed!")



def main():
    cfg = TrainingUnitEncoderConfig_STEP1
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    global_mel_min = torch.load(cfg.dataset.mel_min_path).unsqueeze(-1)
    global_mel_max = torch.load(cfg.dataset.mel_max_path).unsqueeze(-1)


    print(f"Loading filelist from {cfg.dataset.train_filelist_path}")
    validate_channel_wise_mel_normalization(cfg.dataset.train_filelist_path,
                                                          global_mel_min,
                                                          global_mel_max,
                                                          cfg)

    print(f"Loading filelist from {cfg.dataset.test_filelist_path}")
    validate_channel_wise_mel_normalization(cfg.dataset.test_filelist_path,
                                                          global_mel_min,
                                                          global_mel_max,
                                                          cfg)

if __name__ == '__main__':
    main()