import argparse
import json
import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from conf.hydra_config import (
    LJSPeechConfig,
    TrainingUnitEncoderConfig_STEP1,
)
from data import TextMelSpeakerDataset
from unitspeech.duration_predictor import DurationPredictor
from unitspeech.encoder import Encoder
from unitspeech.text import cleaned_text_to_sequence, phonemize, symbols
from unitspeech.unitspeech import UnitSpeech
from unitspeech.util import (
    fix_len_compatibility,
    intersperse,
    load_speaker_embs,
    save_plot,
)
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.models import BigVGAN
from scipy.io.wavfile import write
import phonemizer


logger = logging.getLogger("inference.py")
logger.setLevel(logging.DEBUG)

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig_STEP1)
# cs.store(group="dataset", name="LJSpeech", node=LJSPeechConfig)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig_STEP1):
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
    )
    # Run on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")
    if device.type == "cpu" and cfg.train.on_GPU:
        raise ValueError("CUDA is not available.")
    logger.info(f"Running on: {device.type}")

    # Keep original working directory
    output_dir = os.getcwd()  # Hydra changes the working directory when running the script
    os.chdir(get_original_cwd())
    cfg.train.log_dir = os.path.join(output_dir, cfg.train.log_dir)
    os.makedirs(cfg.train.log_dir, exist_ok=True)

    # Runtime HYPERPARAMS
    num_downsamplings_in_unet = len(cfg.decoder.dim_mults) - 1
    out_size = fix_len_compatibility(
        cfg.train.out_size_second * cfg.data.sampling_rate // cfg.data.hop_length, num_downsamplings_in_unet=num_downsamplings_in_unet
    )

    # Load and initialize Vocoder
    with open(cfg.vocoder.config_path) as f:
        vocoder_hps = AttrDict(json.load(f))
    vocoder = BigVGAN(vocoder_hps)
    vocoder.load_state_dict(torch.load(cfg.vocoder.ckpt_path, map_location=lambda loc, storage: loc)["generator"])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # Load checkpoints for each module
    logger.info("Initializing the Text Encoder...")
    text_encoder = Encoder(
        n_vocab=cfg.encoder.n_vocab,
        n_feats=cfg.data.n_feats,
        n_channels=cfg.encoder.n_channels,
        filter_channels=cfg.encoder.filter_channels,
        n_heads=cfg.encoder.n_heads,
        n_layers=cfg.encoder.n_layers,
        kernel_size=cfg.encoder.kernel_size,
        p_dropout=cfg.encoder.p_dropout,
        window_size=cfg.encoder.window_size,
    ).to(device)
    if not os.path.exists(cfg.encoder.checkpoint):
        raise FileNotFoundError(f"Checkpoint for encoder not found: {cfg.encoder.checkpoint}")
    text_encoder_dict = torch.load(cfg.encoder.checkpoint, map_location=lambda loc, storage: loc)
    text_encoder.load_state_dict(text_encoder_dict["model"])

    logger.info("Initializing the Duration Predictor...")
    duration_predictor = DurationPredictor(
        in_channels=cfg.duration_predictor.in_channels,
        filter_channels=cfg.duration_predictor.filter_channels,
        kernel_size=cfg.duration_predictor.kernel_size,
        p_dropout=cfg.duration_predictor.p_dropout,
        spk_emb_dim=cfg.duration_predictor.spk_emb_dim,
    ).to(device)
    duration_predictor_dict = torch.load(cfg.duration_predictor.checkpoint, map_location=lambda loc, storage: loc)
    duration_predictor.load_state_dict(duration_predictor_dict["model"])

    logger.info("Initializing decoder: GradTTS ...")
    decoder = UnitSpeech(
        n_feats=cfg.data.n_feats,
        dim=cfg.decoder.dim,
        dim_mults=cfg.decoder.dim_mults,
        beta_min=cfg.decoder.beta_min,
        beta_max=cfg.decoder.beta_max,
        pe_scale=cfg.decoder.pe_scale,
        spk_emb_dim=cfg.decoder.spk_emb_dim,
    ).to(device)
    decoder_state_dict = torch.load(cfg.decoder.checkpoint, map_location=lambda loc, storage: loc)
    decoder.load_state_dict(decoder_state_dict["model"])

    logger.info("Loading speaker embeddings")
    pretrained_embs = load_speaker_embs(embs_path=os.path.join(cfg.data.embs_path, cfg.dataset.name), normalize=True)
    speaker_embeddings = torch.nn.Embedding.from_pretrained(pretrained_embs, freeze=True).to(device=device)

    logger.info("Loading spectrogram normalization params")
    mel_max = torch.load(cfg.dataset.mel_max_path).unsqueeze(-1).to(device)
    mel_min = torch.load(cfg.dataset.mel_min_path).unsqueeze(-1).to(device)

    text: str = cfg.inference.text
    SPKR_ID = torch.LongTensor([int(cfg.inference.speaker_id)]).to(device)

    # Process text
    logger.info("Processing text")
    phoneme = phonemize(text, global_phonemizer)
    phoneme = cleaned_text_to_sequence(phoneme)
    phoneme = intersperse(phoneme, len(symbols))  # add a blank token, whose id number is len(symbols)
    phoneme = torch.LongTensor(phoneme).cuda().unsqueeze(0)
    phoneme_lengths = torch.LongTensor([phoneme.shape[-1]]).cuda()

    spk_emb = speaker_embeddings(SPKR_ID).unsqueeze(1).cuda()

    logger.info("Running inference")
    with torch.no_grad():
        y_enc, y_dec, attn = decoder.execute_text_to_speech(
            phoneme=phoneme,
            phoneme_lengths=phoneme_lengths,
            spk_emb=spk_emb,
            text_encoder=text_encoder,
            duration_predictor=duration_predictor,
            num_downsamplings_in_unet=num_downsamplings_in_unet,
            diffusion_steps=cfg.decoder.diffusion_steps,
            length_scale=1.0,
            text_gradient_scale=0.0,
            spk_gradient_scale=0.0,
        )
        save_plot(y_dec.squeeze().cpu(), f"{cfg.train.log_dir}/decoder-normalized.png", title="Mel Spectrogram")
        mel_generated = ((y_dec + 1) / 2 * (mel_max - mel_min) + mel_min)
        audio_generated = vocoder.forward(mel_generated).cpu().squeeze().clamp(-1, 1).numpy()

    save_plot(attn.squeeze().cpu(), f"{cfg.train.log_dir}/attention.png", title="Attention")
    save_plot(y_enc.squeeze().cpu(), f"{cfg.train.log_dir}/encoder.png", title="Encoder")
    save_plot(mel_generated.squeeze().cpu(), f"{cfg.train.log_dir}/decoder-WO-normalization.png", title="Mel Spectrogram")

    write(f"{cfg.train.log_dir}/audio.wav", cfg.data.sampling_rate, audio_generated)


if __name__ == "__main__":
    hydra_main()
