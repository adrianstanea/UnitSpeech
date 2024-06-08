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


count = 0

def get_channel_wise_mel_normalization(filelist_path, text_uncond, cfg: TrainingUnitEncoderConfig_STEP1):
    filelist = parse_filelist(filelist_path, split_char='|')
    global count

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
        crnt_mel_mean = torch.mean(mel, dim=-1, keepdim=True)
        if text_uncond is None:
            text_uncond = crnt_mel_mean
        else:
            # Give equal weight to all samples when computing the mean of the mel spectrogram
            text_uncond = (text_uncond * count + crnt_mel_mean) / (count + 1)
        count += 1
    return text_uncond

def main():
    cfg = TrainingUnitEncoderConfig_STEP1
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    folder_path = os.path.dirname(cfg.dataset.text_uncond_path)
    if not os.path.exists(folder_path):
        print(f"Created directory {folder_path}")
        os.makedirs(folder_path)
    else:
        print(f"Directory {folder_path} already exists")

    text_uncond = None

    try:
        print(f"Loading filelist from {cfg.dataset.test_filelist_path}")
        text_uncond = get_channel_wise_mel_normalization(cfg.dataset.test_filelist_path, text_uncond, cfg)
    except AttributeError:
        print(f"Could not find test_filelist_path in the config file, skipping test_filelist_path")

    try:
        print(f"Loading filelist from {cfg.dataset.train_filelist_path}")
        text_uncond = get_channel_wise_mel_normalization(cfg.dataset.train_filelist_path, text_uncond, cfg)
    except AttributeError:
        print(f"Could not find train_filelist_path in the config file, skipping train_filelist_path")

    try: 
        print(f"Loading filelist from {cfg.dataset.validation_filelist_path}")
        text_uncond = get_channel_wise_mel_normalization(cfg.dataset.validation_filelist_path, text_uncond, cfg)
    except AttributeError:
        print(f"Could not find validation_filelist_path in the config file, skipping validation_filelist_path")

    # Leave the mean unscaled: original code does not scale the mean
    print(f"Saving text_uncond to {cfg.dataset.text_uncond_path}")
    print(f"Global count was: {count}")
    print(f"{text_uncond.shape=}")
    torch.save(text_uncond, cfg.dataset.text_uncond_path)


if __name__ == '__main__':
    main()