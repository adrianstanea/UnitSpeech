""" from https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS """

import json
from math import perm
import os
from typing import Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import phonemizer

from conf.hydra_config import SpeakerEmbedderCfg, UnitExtractorConfig
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN, ECAPA_TDNN_SMALL
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.models import BigVGAN

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0],
                                          [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def fix_len_compatibility(length, num_downsamplings_in_unet=3):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return int(length)
        length += 1


def intersperse(lst, item):
    # Adds blank symbol between each item in the list
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def process_unit(encoded, sampling_rate, hop_length):
    # A method that aligns units and durations (50Hz) extracted from 16kHz audio with
    # mel-spectrograms extracted from 22,050Hz audio.

    unit = encoded["units"].cpu().tolist()
    duration = encoded["durations"].cpu().tolist()

    duration = [int(i) * (sampling_rate // 50) for i in duration]

    expand_unit = []

    for u, d in zip(unit, duration):
        for _ in range(d):
            expand_unit.append(u)

    new_length = len(expand_unit) // hop_length * hop_length

    unit = torch.LongTensor(expand_unit)[
        :new_length].reshape(-1, hop_length).mode(1)[0].tolist()

    squeezed_unit = [unit[0]]
    squeezed_duration = [1]

    for u in unit[1:]:
        if u == squeezed_unit[-1]:
            squeezed_duration[-1] += 1
        else:
            squeezed_unit.append(u)
            squeezed_duration.append(1)

    unit = torch.LongTensor(squeezed_unit)
    duration = torch.LongTensor(squeezed_duration)

    return unit, duration


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss


def save_plot(tensor, savepath, title=None):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    if title:
        plt.title(title)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

def save_for_gif(tensor, savepath, title=None):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def get_phonemizer(language:str):
    if language == 'en-us':
        global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags"
        )
    elif language == 'ro':
        global_phonemizer = phonemizer.backend.EspeakBackend(
            language="ro",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            words_mismatch="ignore",
        )
    else:
        raise ValueError(f"Language {language} not supported.")
    return global_phonemizer

def get_vocoder(config_path, checkpoint, device):
    with open(config_path) as f:
        vocoder_hps = AttrDict(json.load(f))
    vocoder = BigVGAN(vocoder_hps)
    vocoder.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc)["generator"])
    _ = vocoder.to(device).eval()
    vocoder.remove_weight_norm()
    return vocoder

def get_speaker_embedder(device, cfg=SpeakerEmbedderCfg) -> ECAPA_TDNN:
    spk_embedder = ECAPA_TDNN_SMALL(feat_dim=cfg.feat_dim, feat_type=cfg.feat_type, config_path=cfg.config_path)
    state_dict = torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage)
    spk_embedder.load_state_dict(state_dict["model"], strict=False)
    _ = spk_embedder.to(device).eval()
    return spk_embedder

def get_unit_extracter(device, cfg: UnitExtractorConfig):
    unit_extractor = SpeechEncoder.by_name(
        dense_model_name=cfg.dense_model_name,
        quantizer_model_name=cfg.quantizer_name,
        vocab_size=cfg.vocab_size,
        deduplicate=cfg.deduplicate,
        need_f0=cfg.need_f0,
    )
    _ = unit_extractor.to(device).eval()
    return unit_extractor

def load_speaker_embs(embs_path: str, normalize: bool=True) -> Dict[int, torch.Tensor]:
    """Load mean speaker embeddings from .pt files in a directory.

    Args:
        embs_path (str): Folder with .pt files containing speaker embeddings for the current dataset

    Returns:
        torch.Tensor: Tensor containing all speaker embeddings
    """
    embs = {}
    for spk_id_file in sorted(os.listdir(embs_path)):
        if spk_id_file.endswith('.pt'):
            spk_id = int(spk_id_file.split('.')[0])
            spkr_emb = torch.load(os.path.join(embs_path, spk_id_file))

            if normalize:
                spkr_emb = spkr_emb / spkr_emb.norm()

            embs[spk_id] = spkr_emb.squeeze(0)
        else:
            raise ValueError(f"Speaker embedding file {spk_id_file} is not a .pt file.")
    return embs

def random_replace_tensor(spk_embs, replacement_emb, replace_percentage: float=0.25):
    n_items_to_replace = int(spk_embs.size(0) * replace_percentage)
    permutation = torch.randperm(spk_embs.size(0))
    idx_to_replace = permutation[:n_items_to_replace]
    
    for idx in idx_to_replace:
        spk_embs[idx] = replacement_emb
    return spk_embs 


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
