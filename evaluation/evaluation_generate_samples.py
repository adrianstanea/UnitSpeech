import logging
import json
import librosa
import os

# To prevent the path from becoming corrupted when this cell is executed more than once.
try:
    path
except:
    path = "../"
    os.chdir(path)

from numpy import average
import phonemizer
import random
from scipy.io.wavfile import write
import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertModel

from unitspeech.unitspeech import UnitSpeech
from unitspeech.duration_predictor import DurationPredictor
from unitspeech.encoder import Encoder
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
from unitspeech.text import cleaned_text_to_sequence, phonemize, symbols
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import HParams, fix_len_compatibility, get_speaker_embedder, get_unit_extracter, get_vocoder, intersperse, process_unit, generate_path, sequence_mask
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.meldataset import mel_spectrogram
from unitspeech.vocoder.models import BigVGAN
import time

from conf.hydra_config import (
    MainConfig,
)
import pandas as pd

import soundfile as sf

from unitspeech.util import (
    fix_len_compatibility,
    save_plot,
    sequence_mask,
)
import numpy as np

# COMPTUATIONS
processing_time = np.array([])
speech_duration = np.array([])


cfg = MainConfig
device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

print(f"Running from {os.getcwd()}")
print(f"Device: {device}")

column_names = ["path", "transcript", "speaker_id"]
reference_speech_samples = pd.read_csv("evaluation/reference_speech_samples.csv", delimiter="|", header=None, names=column_names)
eval_speech_samples = pd.read_csv("evaluation/evaluation.csv", delimiter="|", header=None, names=column_names)


finetune_config_path = "unitspeech/checkpoints/finetune.json"
with open(finetune_config_path, "r") as f:
    data = f.read()
finetune_config = json.loads(data)


fp16_run = False
learning_rate = 2e-5
# Runtime HYPERPARAMS
num_downsamplings_in_unet = len(cfg.decoder.dim_mults) - 1
out_size = fix_len_compatibility(
    cfg.train.out_size_second * cfg.data.sampling_rate // cfg.data.hop_length,
    num_downsamplings_in_unet=num_downsamplings_in_unet,
)

hps_finetune = HParams(**finetune_config)

speaker_encoder_path = "/checkpoints/EVALUATION/speaker_encoder/checkpts/speaker_encoder.pt"

print("Initializing Vocoder...")
vocoder = get_vocoder(config_path=cfg.vocoder.config_path, checkpoint=cfg.vocoder.ckpt_path, device=device)


# Speaker Encoder for extracting speaker embedding
print("Initializing Speaker Encoder...")
spk_embedder = get_speaker_embedder(device, cfg.spkr_embedder)


# Unit Extractor for extraction unit and duration, which are used for finetuning
print("Initializing Unit Extracter...")
# unit_extractor = SpeechEncoder.by_name(
#     dense_model_name=cfg.unit_extractor.dense_model_name,
#     quantizer_model_name=cfg.unit_extractor.quantizer_name,
#     vocab_size=cfg.unit_extractor.vocab_size,
#     deduplicate=cfg.unit_extractor.deduplicate,
#     need_f0=cfg.unit_extractor.need_f0,
# )
# _ = unit_extractor.cuda().eval()
unit_extractor = get_unit_extracter(device, cfg.unit_extractor)

text_encoder = Encoder(
    n_vocab=cfg.text_encoder.n_vocab,
    n_feats=cfg.data.n_feats,
    n_channels=cfg.text_encoder.n_channels,
    filter_channels=cfg.text_encoder.filter_channels,
    n_heads=cfg.text_encoder.n_heads,
    n_layers=cfg.text_encoder.n_layers,
    kernel_size=cfg.text_encoder.kernel_size,
    p_dropout=cfg.text_encoder.p_dropout,
    window_size=cfg.text_encoder.window_size,
).to(device)
if not os.path.exists(cfg.text_encoder.checkpoint):
    raise FileNotFoundError(f"Checkpoint for encoder not found: {cfg.text_encoder.checkpoint}")
text_encoder_dict = torch.load(cfg.text_encoder.checkpoint, map_location=lambda loc, storage: loc)
text_encoder.load_state_dict(text_encoder_dict["model"])
_ = text_encoder.eval()

duration_predictor = DurationPredictor(
    in_channels=cfg.duration_predictor.in_channels,
    filter_channels=cfg.duration_predictor.filter_channels,
    kernel_size=cfg.duration_predictor.kernel_size,
    p_dropout=cfg.duration_predictor.p_dropout,
    spk_emb_dim=cfg.duration_predictor.spk_emb_dim,
)
if not os.path.exists(cfg.duration_predictor.checkpoint):
    raise FileNotFoundError(f"Checkpoint for duration predictor not found: {cfg.duration_predictor.checkpoint}")
duration_predictor_dict = torch.load(cfg.duration_predictor.checkpoint, map_location=lambda loc, storage: loc)
duration_predictor.load_state_dict(duration_predictor_dict["model"])
_ = duration_predictor.cuda().eval()

unit_encoder_path = "unitspeech/checkpoints/unit_encoder.pt"
unit_encoder = Encoder(
    n_vocab=cfg.data.n_units,
    n_feats=cfg.data.n_feats,
    n_channels=cfg.text_encoder.n_channels,
    filter_channels=cfg.text_encoder.filter_channels,
    n_heads=cfg.text_encoder.n_heads,
    n_layers=cfg.text_encoder.n_layers,
    kernel_size=cfg.text_encoder.kernel_size,
    p_dropout=cfg.text_encoder.p_dropout,
    window_size=cfg.text_encoder.window_size,
).to(device)
unit_encoder_dict = torch.load(unit_encoder_path, map_location=lambda storage, loc: storage)
unit_encoder.load_state_dict(unit_encoder_dict["model"])
_ = unit_encoder.eval()

# Normalization parameters for mel spectrogram
decoder_dict = torch.load(cfg.decoder.checkpoint, map_location=lambda loc, storage: loc)

mel_max = decoder_dict["mel_max"]
mel_min = decoder_dict["mel_min"]

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="ro",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    words_mismatch="ignore",
)

# The text gradient scale is responsible for pronunciation and audio quality.
# The default value is 1, and increasing the value improves pronunciation accuracy but may reduce speaker similarity.
# We recommend starting with 0 and gradually increasing it if the pronunciation is not satisfactory.
text_gradient_scale = 1.0

# The speaker gradient scale is responsible for speaker similarity.
# Increasing the value enhances speaker similarity but may slightly degrade pronunciation and audio quality.
# For unique voices, we recommend using a larger value for the speaker gradient scale.
spk_gradient_scale = 1.0

# We have confirmed that our duration predictor is not accurately following the duration of the reference audio as expected.
# As a result, while the reference audio's tone and speaking style are well adapted, there are differences in speech rate.
# To address this issue, we use the "length_scale" argument as in Grad-TTS to mitigate the discrepancy.
# If the value of "length_scale" is greater than 1, the speech rate will be slower.
# Conversely, if the value is less than 1, the speech rate will be faster.
length_scale = 1.0

# The number of diffusion steps during sampling refers to the number of iterations performed to improve audio quality.
# Generally, larger values lead to better audio quality but slower sampling speeds.
# Conversely, smaller values allow for faster sampling but may result in lower audio quality.
diffusion_steps = 500


for outer_idx, outer_row in reference_speech_samples.iterrows():
    # =====================================================================================
    path = outer_row["path"]
    transcript = outer_row["transcript"]
    speaker_id = outer_row["speaker_id"]
    print(f"ID: {speaker_id}")
    # =====================================================================================
    # FINETUNE
    # Diffusion-based acoutstic model to be finetuned to the current speaker
    decoder = UnitSpeech(
        n_feats=cfg.data.n_feats,
        dim=cfg.decoder.dim,
        dim_mults=cfg.decoder.dim_mults,
        beta_min=cfg.decoder.beta_min,
        beta_max=cfg.decoder.beta_max,
        pe_scale=cfg.decoder.pe_scale,
        spk_emb_dim=cfg.decoder.spk_emb_dim,
    ).to(device)
    decoder_dict = torch.load(cfg.decoder.checkpoint, map_location=lambda loc, storage: loc)
    decoder.load_state_dict(decoder_dict["model"])
    _ = decoder.train()

    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=learning_rate)
    if fp16_run:
        scaler = torch.cuda.amp.GradScaler()

    # DATA PROCESSING
    wav, sr = librosa.load(path)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    mel = mel_spectrogram(
        wav,
        cfg.data.n_fft,
        cfg.data.n_feats,
        cfg.data.sampling_rate,
        cfg.data.hop_length,
        cfg.data.win_length,
        cfg.data.mel_fmin,
        cfg.data.mel_fmax,
        center=False,
    )
    mel_max = decoder_dict["mel_max"]
    mel_min = decoder_dict["mel_min"]
    mel = (mel - mel_min) / (mel_max - mel_min) * 2 - 1
    mel = mel.cuda()
    # Speaker embedder expects 16KHz audio samples
    resample_fn = torchaudio.transforms.Resample(sr, cfg.spkr_embedder.sr).cuda()
    wav = resample_fn(wav.cuda())
    spk_emb = spk_embedder(wav)
    # User speaker embeddings with norm = 1
    spk_emb = spk_emb / spk_emb.norm()
    # Extract the units and unit durations to be used for fine-tuning.
    encoded = unit_extractor(wav.to("cuda"))  # => units with f_unit freq: 16Khz
    # Upsample unit and durations from f_unit to f_mel
    unit, duration = process_unit(encoded, cfg.spkr_embedder.sr, cfg.data.hop_length)
    # Reshape the input to match the dimensions and convert it to a PyTorch tensor.
    unit = unit.unsqueeze(0).cuda()
    duration = duration.unsqueeze(0).cuda()
    mel = mel.cuda()
    unit_lengths = torch.LongTensor([unit.shape[-1]]).cuda()
    mel_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
    spk_emb = spk_emb.cuda().unsqueeze(1)
    # Prepare unit encoder output for finetuning
    with torch.no_grad():
        cond_x, x, x_mask = unit_encoder(unit, unit_lengths)
    mel_max_length = mel.shape[-1]
    mel_mask = sequence_mask(mel_lengths, mel_max_length).unsqueeze(1).to(x_mask)
    attn_mask = x_mask.unsqueeze(-1) * mel_mask.unsqueeze(2)
    attn = generate_path(duration, attn_mask.squeeze(1))

    # Finetune the decoder
    for _ in tqdm(range(cfg.finetune.steps)):
        cond_x = cond_x.detach()
        mel = mel.detach()
        mel_mask = mel_mask.detach()
        mel_lengths = mel_lengths.detach()
        spk_emb = spk_emb.detach()
        attn = attn.detach()

        decoder.zero_grad()

        with torch.cuda.amp.autocast(enabled=fp16_run):
            diff_loss = decoder.fine_tune(
                cond_x,
                mel,
                mel_mask,
                mel_lengths,
                mel_max_length,
                attn,
                spk_emb,
                out_size,
                cfg.data.n_feats,
            )

        loss = sum([diff_loss])

        if fp16_run:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
            optimizer.step()
    # =====================================================================================
    # NOW USE THE FINETUNED DECODER FOR INFERENCE OVER THE EVALUATION SAMPLES
    eval_samples_crnt_speaker = eval_speech_samples[eval_speech_samples["speaker_id"] == speaker_id]
    print(eval_samples_crnt_speaker["speaker_id"].value_counts())

    # Load the normalization parameters for mel-spectrogram normalization.
    mel_max = decoder_dict["mel_max"].cuda()
    mel_min = decoder_dict["mel_min"].cuda()

    base_path = "/outputs/evaluation/with-finetune_AWGN_500"
    for inner_idx, inner_row in eval_samples_crnt_speaker.iterrows():
        inner_path = inner_row["path"]
        # Get the sample name from inner_path
        crnt_sample = inner_path.split("/")[-1].split(".")[0]  # no file extension

        inner_transcript = inner_row["transcript"]
        inner_speaker_id = inner_row["speaker_id"]
        assert (
            inner_speaker_id == speaker_id
        ), "Running inference on a different speaker ID than current finetuned model."
        print(f"Inference on speaker ID {inner_speaker_id}")
        print(f"\tPath: {inner_path}")
        print(f"\tTranscript: {inner_transcript}")

        # Metrics for future RTF
        start_time = time.time()
        phoneme = phonemize(inner_transcript, global_phonemizer)
        phoneme = cleaned_text_to_sequence(phoneme)
        phoneme = intersperse(phoneme, len(symbols))
        phoneme = torch.LongTensor(phoneme).cuda().unsqueeze(0)
        phoneme_lengths = torch.LongTensor([phoneme.shape[-1]]).cuda()

        with torch.no_grad():
            y_enc, y_dec, _attn = decoder.execute_text_to_speech(
                phoneme=phoneme,
                phoneme_lengths=phoneme_lengths,
                spk_emb=spk_emb,
                text_encoder=text_encoder,
                duration_predictor=duration_predictor,
                num_downsamplings_in_unet=num_downsamplings_in_unet,
                diffusion_steps=diffusion_steps,
                length_scale=length_scale,
                text_gradient_scale=text_gradient_scale,
                spk_gradient_scale=spk_gradient_scale,
            )
            mel_generated = (y_dec + 1) / 2 * (mel_max - mel_min) + mel_min
            synthesized_audio = vocoder.forward(mel_generated).cpu().squeeze().clamp(-1, 1).numpy()

            end_time = time.time()
            T_processing = end_time - start_time

            write(f"{base_path}/{crnt_sample}.wav", cfg.data.sampling_rate, synthesized_audio)

            speech_audio, loaded_sr = librosa.load(f"{base_path}/{crnt_sample}.wav", sr=None)
            T_speech = librosa.get_duration(y=speech_audio, sr=loaded_sr)

            processing_time = np.append(processing_time, T_processing)
            speech_duration = np.append(speech_duration, T_speech)


np.save("processing_time_with-finetune_AWGN_500.npy", processing_time)
np.save("speech_duration_with-finetune_AWGN_500.npy", speech_duration)