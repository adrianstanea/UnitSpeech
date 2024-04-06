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

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from unitspeech.util import fix_len_compatibility, parse_filelist, process_unit, intersperse

from unitspeech.vocoder.meldataset import mel_spectrogram


import random
import librosa
import numpy as np

import torch
import torchaudio as ta

from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder

# class TextMelDataset(torch.utils.data.Dataset):
#     def __init__(self,
#                  filelist_path,
#                  cmudict_path,
#                  random_seed=42,
#                  add_blank=True,
#                  n_fft=1024,
#                  n_mels=80,
#                  sample_rate=22050,
#                  hop_length=256,
#                  win_length=1024,
#                  f_min=0.,
#                  f_max=8000):
#         self.filepaths_and_text = parse_filelist(filelist_path)
#         self.cmudict = cmudict.CMUDict(cmudict_path)
#         self.add_blank = add_blank
#         self.n_fft = n_fft
#         self.n_mels = n_mels
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.f_min = f_min
#         self.f_max = f_max
#         random.seed(random_seed)
#         random.shuffle(self.filepaths_and_text)

#     def get_pair(self, filepath_and_text):
#         filepath, text = filepath_and_text[0], filepath_and_text[1]
#         text = self.get_text(text, add_blank=self.add_blank)
#         mel = self.get_mel(filepath)
#         return (text, mel)

#     def get_mel(self, filepath):
#         audio, sr = ta.load(filepath)
#         assert sr == self.sample_rate
#         mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
#                               self.win_length, self.f_min, self.f_max, center=False).squeeze()
#         return mel

#     def get_text(self, text, add_blank=True):
#         text_norm = text_to_sequence(text, dictionary=self.cmudict)
#         if self.add_blank:
#             text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
#         text_norm = torch.IntTensor(text_norm)
#         return text_norm

#     def __getitem__(self, index):
#         text, mel = self.get_pair(self.filepaths_and_text[index])
#         item = {'y': mel, 'x': text}
#         return item

#     def __len__(self):
#         return len(self.filepaths_and_text)

#     def sample_test_batch(self, size):
#         idx = np.random.choice(range(len(self)), size=size, replace=False)
#         test_batch = []
#         for index in idx:
#             test_batch.append(self.__getitem__(index))
#         return test_batch


# class TextMelBatchCollate(object):
#     def __call__(self, batch):
#         B = len(batch)
#         y_max_length = max([item['y'].shape[-1] for item in batch])
#         y_max_length = fix_len_compatibility(y_max_length)
#         x_max_length = max([item['x'].shape[-1] for item in batch])
#         n_feats = batch[0]['y'].shape[-2]

#         y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
#         x = torch.zeros((B, x_max_length), dtype=torch.long)
#         y_lengths, x_lengths = [], []

#         for i, item in enumerate(batch):
#             y_, x_ = item['y'], item['x']
#             y_lengths.append(y_.shape[-1])
#             x_lengths.append(x_.shape[-1])
#             y[i, :, :y_.shape[-1]] = y_
#             x[i, :x_.shape[-1]] = x_

#         y_lengths = torch.LongTensor(y_lengths)
#         x_lengths = torch.LongTensor(x_lengths)
#         return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}

class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filelist_path,
                 cmudict_path,
                 spk_embedder,
                 random_seed=42,
                 add_blank=True,
                 n_fft=1024,
                 n_mels=80,
                 sample_rate=22050,
                 hop_length=256,
                 win_length=1024,
                 f_min=0.,
                 f_max=8000,
                 device='cuda',
                 load_preprocessed=False):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.spk_embedder = spk_embedder
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank

        self.device=device
        self.load_preprocessed = load_preprocessed
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text = line[0], line[1]

        if self.load_preprocessed:
            base_dir = os.path.dirname(filepath)

            text = self.get_text(text, add_blank=self.add_blank)

            mel_path = os.path.join(base_dir, 'mels', os.path.basename(filepath).replace('.wav', '.npy'))

            if not os.path.exists(mel_path):
                raise FileNotFoundError(f"Mel spectrogram file not found: {mel_path}. Please run the process_dataset script.")
            mel = torch.from_numpy(np.load(mel_path))

            spk_emb_path = os.path.join(base_dir, 'speaker_embeddings', os.path.basename(filepath).replace('.wav', '.npy'))
            if not os.path.exists(spk_emb_path):
                raise FileNotFoundError(f"Speaker embedding file not found: {spk_emb_path}. Please run the process_dataset script.")
            spk_emb = torch.from_numpy(np.load(spk_emb_path))
        else:
            text = self.get_text(text, add_blank=self.add_blank)
            mel = self.get_mel(filepath)
            spk_emb = self.get_speaker_emb(filepath)

        return (text, mel, spk_emb)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        # print(f"Read sample rate: {sr}")
        # print(f"Configured sample rate: {self.sample_rate}")
        # TODO: change global config sample rate or adapt with a resampler
        # assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker_emb(self, filepath):
        # TODO: this is a hard bottleneck !!!! -> takes to long to laod data
        audio, sr = ta.load(filepath)

        # TODO: in UnitSpeech the was is resampled to 16Khz before extracting embedding
        # resample_fn = ta.transforms.Resample(sr, 16_000).to(self.device)
        # audio = resample_fn(audio.to(self.device))
        # resample_fn = ta.transforms.Resample(sr, 16_000)
        # audio = resample_fn(audio)

        spk_emb = self.spk_embedder(audio.to(self.device))
        # TODO: should the spk embedding be normalized at train time???
        # spk_emb = spk_emb / spk_emb.norm()
        return spk_emb.cpu()
        # return spk_emb.clone().detach()

    def __getitem__(self, index):
        text, mel, spk_emb = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk_emb': spk_emb}
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
        spk_emb = []

        for i, item in enumerate(batch):
            y_, x_, spk_emb_ = item['y'], item['x'], item['spk_emb']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk_emb.append(spk_emb_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk_emb = torch.cat(spk_emb, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk_emb': spk_emb}



# class TextMelSpeakerDataset(torch.utils.data.Dataset):
#     def __init__(self,
#                  filelist_path,
#                  cmudict_path,
#                  random_seed=42,
#                  add_blank=True,
#                  n_fft=1024,
#                  n_mels=80,
#                  sample_rate=22050,
#                  hop_length=256,
#                  win_length=1024,
#                  f_min=0.,
#                  f_max=8000):
#         super().__init__()
#         self.filelist = parse_filelist(filelist_path, split_char='|')
#         self.cmudict = cmudict.CMUDict(cmudict_path)
#         self.n_fft = n_fft
#         self.n_mels = n_mels
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.f_min = f_min
#         self.f_max = f_max
#         self.add_blank = add_blank
#         random.seed(random_seed)
#         random.shuffle(self.filelist)

#     def get_triplet(self, line):
#         filepath, text, speaker = line[0], line[1], line[2]
#         text = self.get_text(text, add_blank=self.add_blank)
#         mel = self.get_mel(filepath)
#         speaker = self.get_speaker(speaker)
#         return (text, mel, speaker)

#     def get_mel(self, filepath):
#         audio, sr = ta.load(filepath)
#         assert sr == self.sample_rate
#         mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
#                               self.win_length, self.f_min, self.f_max, center=False).squeeze()
#         return mel

#     def get_text(self, text, add_blank=True):
#         text_norm = text_to_sequence(text, dictionary=self.cmudict)
#         if self.add_blank:
#             text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
#         text_norm = torch.LongTensor(text_norm)
#         return text_norm

#     def get_speaker(self, speaker):
#         speaker = torch.LongTensor([int(speaker)])
#         return speaker

#     def __getitem__(self, index):
#         text, mel, speaker = self.get_triplet(self.filelist[index])
#         item = {'y': mel, 'x': text, 'spk': speaker}
#         return item

#     def __len__(self):
#         return len(self.filelist)

#     def sample_test_batch(self, size):
#         idx = np.random.choice(range(len(self)), size=size, replace=False)
#         test_batch = []
#         for index in idx:
#             test_batch.append(self.__getitem__(index))
#         return test_batch


# class TextMelSpeakerBatchCollate(object):
#     def __call__(self, batch):
#         B = len(batch)
#         y_max_length = max([item['y'].shape[-1] for item in batch])
#         y_max_length = fix_len_compatibility(y_max_length)
#         x_max_length = max([item['x'].shape[-1] for item in batch])
#         n_feats = batch[0]['y'].shape[-2]

#         y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
#         x = torch.zeros((B, x_max_length), dtype=torch.long)
#         y_lengths, x_lengths = [], []
#         spk = []

#         for i, item in enumerate(batch):
#             y_, x_, spk_ = item['y'], item['x'], item['spk']
#             y_lengths.append(y_.shape[-1])
#             x_lengths.append(x_.shape[-1])
#             y[i, :, :y_.shape[-1]] = y_
#             x[i, :x_.shape[-1]] = x_
#             spk.append(spk_)

#         y_lengths = torch.LongTensor(y_lengths)
#         x_lengths = torch.LongTensor(x_lengths)
#         spk = torch.cat(spk, dim=0)
#         return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}



# class UnitMelSpeakerDataset(torch.utils.data.Dataset):
#     def __init__(self,
#                  filelist_path,
#                  unit_extractor: SpeechEncoder,
#                  n_fft=1024,
#                  n_mels=80,
#                  sample_rate=22050,
#                  hop_length=256,
#                  win_length=1024,
#                  f_min=0.,
#                  f_max=8000,
#                  random_seed=42):
#         super().__init__()
#         self.filelist = parse_filelist(filelist_path, split_char='|')
#         self.n_fft = n_fft
#         self.n_mels = n_mels
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.f_min = f_min
#         self.f_max = f_max

#         self.random_seed = random_seed
#         random.seed(random_seed)
#         random.shuffle(self.filelist)

#         if not isinstance(unit_extractor, SpeechEncoder):
#             raise ValueError("Unit extractor is required.")
#         self.unit_extractor = unit_extractor

#     def get_triplet(self, line):
#         filepath, text, speaker_id = line[0], line[1], line[2]
#         unit = self.get_unit()
#         mel = self.get_mel(filepath)
#         speaker_id = self.get_speaker(speaker_id)
#         return (unit, mel, speaker_id)

#     def get_unit(self, filepath):
#         wav, sr = librosa.load(filepath, sr=self.sample_rate)
#         wav = torch.FloatTensor(wav.to("cuda")) # .unsqueeze(0) # Add batch dimension? Is it needed if we are dataloader?
#         encoded = self.unit_extractor(wav)
#         unit, duration = process_unit(encoded, self.sample_rate, self.hop_length)

#     def get_mel(self, filepath):
#         # TODO: does it matter if we load with librosa or with torchaudio?
#         # TODO: do we add normalization here to mel_spectrogram?
#         wav, sr = librosa.load(filepath, sr=self.sample_rate)
#         mel = mel_spectrogram(y=wav,
#                               n_fft=self.n_fft,
#                               num_mels=self.n_mels,
#                               sampling_rate=self.sample_rate,
#                               hop_size=self.hop_length,
#                               win_size=self.win_length,
#                               fmin=self.f_min,
#                               fmax=self.f_max)
#         return mel

#     def get_speaker(self, speaker):
#         speaker = torch.LongTensor([int(speaker)])
#         return speaker

#     def __getitem__(self, index):
#         unit, mel, speaker = self.get_triplet(self.filelist[index])
#         item = {'y': mel, 'x': unit, 'spkr': speaker}
#         return item

#     def __len__(self):
#         return len(self.filelist)

#     def sample_test_batch(self, size):
#         idx = np.random.choice(range(len(self)), size=size, replace=False)
#         test_batch = []
#         for index in idx:
#             test_batch.append(self.__getitem__(index))
#         return test_batch

# # TODO: combe back and adapt for units !!


# class UnitMelSpeakerBatchCollate(object):
#     def __call__(self, batch):
#         B = len(batch)
#         y_max_length = max([item['y'].shape[-1] for item in batch])
#         # y_max_length = fix_len_compatibility(y_max_length)
#         x_max_length = max([item['x'].shape[-1] for item in batch])
#         n_feats = batch[0]['y'].shape[-2]

#         y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
#         x = torch.zeros((B, x_max_length), dtype=torch.long)
#         y_lengths, x_lengths = [], []
#         spk = []

#         for i, item in enumerate(batch):
#             y_, x_, spk_ = item['y'], item['x'], item['spk']
#             y_lengths.append(y_.shape[-1])
#             x_lengths.append(x_.shape[-1])
#             y[i, :, :y_.shape[-1]] = y_
#             x[i, :x_.shape[-1]] = x_
#             spk.append(spk_)

#         y_lengths = torch.LongTensor(y_lengths)
#         x_lengths = torch.LongTensor(x_lengths)
#         spk = torch.cat(spk, dim=0)
#         return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}