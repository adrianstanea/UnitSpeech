# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import logging
import os
import random

import numpy as np
import phonemizer
import phonemizer.logger
import torch
import torchaudio as ta

# import librosa
from phonemizer.logger import get_logger

from unitspeech.text import cleaned_text_to_sequence, phonemize, symbols
from unitspeech.util import (
    fix_len_compatibility,
    intersperse,
    parse_filelist,
)
from unitspeech.vocoder.meldataset import mel_spectrogram


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filelist_path,
                 random_seed=42,
                 add_blank=True,
                 n_fft=1024,
                 n_mels=80,
                 sample_rate=22050,
                 hop_length=256,
                 win_length=1024,
                 f_min=0.,
                 f_max=8000,
                 normalize_mels=True,
                 mel_min_path=None,
                 mel_max_path=None,
                 ):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank

        random.seed(random_seed)
        random.shuffle(self.filelist)

        self.normalize_mels = normalize_mels
        if normalize_mels:
            assert mel_min_path is not None, "Mel min path is required for normalization"
            assert mel_max_path is not None, "Mel max path is required for normalization"
            self.mel_min = torch.load(mel_min_path).unsqueeze(-1)
            self.mel_max= torch.load(mel_max_path).unsqueeze(-1)

        phonemizer_logger = get_logger(verbosity='quiet')
        # TODO: check if the mismatch warnining on WARNING log level is relevant
        # https://github.com/lifeiteng/vall-e/issues/5 => CONFIRMS IT SHOULD BE FINE
        phonemizer_logger.setLevel(logging.ERROR)
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us',
                                                                  preserve_punctuation=True,
                                                                  with_stress=True,
                                                                  words_mismatch='ignore',
                                                                  logger=phonemizer_logger)

    def get_triplet(self, line):
        filepath, text, speaker_id = line[0], line[1], line[2]
        mel = self.get_mel(filepath)
        text = self.get_text(text,add_blank=self.add_blank)
        spk_id = self.get_speaker_id(speaker_id)
        return (text, mel, spk_id)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)

        if sr != self.sample_rate:
            resample_fn = ta.transforms.Resample(sr, self.sample_rate)
            audio = resample_fn(audio)

        mel = mel_spectrogram(audio,
                              self.n_fft,
                              self.n_mels,
                              self.sample_rate,
                              self.hop_length,
                              self.win_length,
                              self.f_min,
                              self.f_max,
                              center=False).squeeze()
        if self.normalize_mels:
            mel = (mel - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1 # Normalize mel-spectrogram: [-1, 1]
        return mel

    def get_text(self, text, add_blank=True):
        # Use same phonemizer as it was done with UnitSpeech, original GradTTS uses CMUDict
        phoneme = phonemize(text, self.global_phonemizer)
        phoneme = cleaned_text_to_sequence(phoneme)
        if add_blank:
             # add a blank token, whose id number is len(symbols)
            phoneme = intersperse(phoneme, len(symbols))
        phoneme = torch.LongTensor(phoneme)
        return phoneme

    def get_speaker_id(self, speaker):
        # use key to extract speaker embedding from a dictionary
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, spk_id = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk_id': spk_id}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk_id = []

        for i, item in enumerate(batch):
            y_, x_, spk_id_ = item['y'], item['x'], item['spk_id']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            # Copy data an pad with 0 up to max length
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk_id.append(spk_id_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk_id = torch.cat(spk_id, dim=0)
        return {'x': x,
                'x_lengths': x_lengths,
                'y': y,
                'y_lengths': y_lengths,
                'spk_id': spk_id
                }


class UnitDurationMelSPeakerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filelist_path,
                 random_seed=42,
                 add_blank=True,
                 n_fft=1024,
                 n_mels=80,
                 sample_rate=22050,
                 hop_length=256,
                 win_length=1024,
                 f_min=0.,
                 f_max=8000,
                 normalize_mels=True,
                 mel_min_path=None,
                 mel_max_path=None,
                 ):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank

        random.seed(random_seed)
        random.shuffle(self.filelist)

        self.normalize_mels = normalize_mels
        if normalize_mels:
            assert mel_min_path is not None, "Mel min path is required for normalization"
            assert mel_max_path is not None, "Mel max path is required for normalization"
            self.mel_min = torch.load(mel_min_path).unsqueeze(-1)
            self.mel_max= torch.load(mel_max_path).unsqueeze(-1)

    def get_quadruple(self, line):
        filepath, _, speaker_id = line[0], line[1], line[2]

        unit, duration = self.get_unit_duration(filepath)
        mel = self.get_mel(filepath)
        spk_id = self.get_speaker_id(speaker_id)
        return (unit, duration, mel, spk_id)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)

        if sr != self.sample_rate:
            resample_fn = ta.transforms.Resample(sr, self.sample_rate)
            audio = resample_fn(audio)

        mel = mel_spectrogram(audio,
                              self.n_fft,
                              self.n_mels,
                              self.sample_rate,
                              self.hop_length,
                              self.win_length,
                              self.f_min,
                              self.f_max,
                              center=False).squeeze()
        if self.normalize_mels:
            mel = (mel - self.mel_min) / (self.mel_max - self.mel_min) * 2 - 1 # Normalize mel-spectrogram: [-1, 1]
        return mel

    def get_unit_duration(self, filepath):
        # Loads preprocessed units and durations scaled to 22.05KHz
        parent_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        basename, extension = os.path.splitext(filename)

        unit_base = f"{basename}_unit.pt"
        duration_base = f"{basename}_duration.pt"

        unit = torch.load(os.path.join(parent_dir, unit_base))
        duration = torch.load(os.path.join(parent_dir, duration_base))
        return (unit, duration)

    def get_speaker_id(self, speaker):
        # use key to extract speaker embedding from a dictionary
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        unit, duration, mel, spk_id = self.get_quadruple(self.filelist[index])
        item = {'y': mel,
                'x': unit,
                'x_duration': duration,
                'spk_id': spk_id}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class UnitDurationMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        x_duration_max_length = max([item['x_duration'].shape[-1] for item in batch])
        assert x_max_length == x_duration_max_length, "Unit and durations should have same shape"
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        x_duration = torch.zeros((B, x_duration_max_length), dtype=torch.long)
        y_lengths, x_lengths, x_duration_lengths = [], [], []
        spk_id = []

        for i, item in enumerate(batch):
            y_, x_, x_duration_, spk_id_ = item['y'], item['x'], item['x_duration'], item['spk_id']
            # Track original lengths before batch
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            x_duration_lengths.append(x_duration_.shape[-1])
            # Save values and pad with 0 up to max length
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            x_duration[i, :x_duration_.shape[-1]] = x_duration_
            spk_id.append(spk_id_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        x_duration_lengths = torch.LongTensor(x_duration_lengths)
        spk_id = torch.cat(spk_id, dim=0)
        return {'x': x, 'x_lengths': x_lengths,
                'x_duration': x_duration, 'x_duration_lengths': x_duration_lengths,
                'y': y, 'y_lengths': y_lengths,
                'spk_id': spk_id}