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
from unitspeech.util import load_speaker_embs, parse_filelist
from unitspeech.vocoder.meldataset import mel_spectrogram


def main():
    cfg = TrainingUnitEncoderConfig_STEP1
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    folder_path = os.path.dirname(cfg.dataset.spk_uncond_path)
    if not os.path.exists(folder_path):
        print(f"Created directory {folder_path}")
        os.makedirs(folder_path)
    else:
        print(f"Directory {folder_path} already exists")

    spk_uncond = None
    pretrained_embs = load_speaker_embs(embs_path =os.path.join(cfg.data.embs_path, cfg.dataset.name),
                                        normalize=False)
    pretrained_embs = torch.stack(list(pretrained_embs.values()), dim=0)
    print(f"{pretrained_embs.shape=}")
    # resize to network shape (1, 1, 256)
    spk_uncond = torch.mean(pretrained_embs, dim=0, keepdim=True).unsqueeze(0) 
    print(f"{spk_uncond.shape=}")

    print(f"Saving spk_uncond to {cfg.dataset.spk_uncond_path}")
    torch.save(spk_uncond , cfg.dataset.spk_uncond_path)


if __name__ == '__main__':
    main()