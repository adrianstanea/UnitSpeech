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
from unitspeech.unitspeech import UnitSpeech
from unitspeech.util import (
    fix_len_compatibility,
    load_speaker_embs,
    save_plot,
)
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.models import BigVGAN
from scipy.io.wavfile import write


logger = logging.getLogger("inference.py")
logger.setLevel(logging.DEBUG)

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig_STEP1)
cs.store(group="dataset", name="LJSpeech", node=LJSPeechConfig)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig_STEP1):
    text: str = "Hello, my name is Bogdan. I am creating a demonstration sample."
    
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

    # Load samples for inference
    logger.info("Loading validation dataset...")
    test_dataset = TextMelSpeakerDataset(
        filelist_path=cfg.dataset.test_filelist_path,
        random_seed=cfg.train.seed,
        add_blank=cfg.data.add_blank,
        n_fft=cfg.data.n_fft,
        n_mels=cfg.data.n_feats,
        sample_rate=cfg.data.sampling_rate,
        hop_length=cfg.data.hop_length,
        win_length=cfg.data.win_length,
        f_min=cfg.data.mel_fmin,
        f_max=cfg.data.mel_fmax,
        normalize_mels=cfg.dataset.normalize_mels,
        mel_min_path=cfg.dataset.mel_min_path,
        mel_max_path=cfg.dataset.mel_max_path,
    )
    test_batch = test_dataset.sample_test_batch(size=1)  # Change to load more samples

    logger.info("Running inference over test samples")
    with torch.no_grad() as no_grad, torch.autograd.set_detect_anomaly(True) as anomaly_detect:
        for idx, item in enumerate(test_batch):
            logger.info(f"\tProcessing sample {idx}")
            x = item["x"].to(torch.long).unsqueeze(0).cuda()  # phonemes: (batch, phonemes)
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()  # phoneme lengths: (batch)
            spk_id = item["spk_id"].cuda()  # (batch)
            spk_embs = speaker_embeddings(spk_id).unsqueeze(1).cuda()

            y_enc, y_dec, attn = decoder.execute_text_to_speech(
                phoneme=x,
                phoneme_lengths=x_lengths,
                spk_emb=spk_embs,
                text_encoder=text_encoder,
                duration_predictor=duration_predictor,
                num_downsamplings_in_unet=num_downsamplings_in_unet,
                diffusion_steps=cfg.decoder.diffusion_steps,
                length_scale=1.0,
                text_gradient_scale=0.0,
                spk_gradient_scale=0.0,
            )
            # Scale back the generated mel = y_dec
            save_plot(y_dec.squeeze().cpu(), f"{cfg.train.log_dir}/decoder-normalized_{idx}.png", title="Mel Spectrogram")

            mel_generated = (y_dec + 1) / 2 * (mel_max - mel_min)
            audio_generated = vocoder.forward(mel_generated).cpu().squeeze().clamp(-1, 1).numpy()

            # OPTIONAL: save outputs of the current run
            save_plot(attn.squeeze().cpu(), f"{cfg.train.log_dir}/attention_{idx}.png", title="Attention")
            save_plot(y_enc.squeeze().cpu(), f"{cfg.train.log_dir}/encoder_{idx}.png", title="Encoder")
            save_plot(y_dec.squeeze().cpu(), f"{cfg.train.log_dir}/decoder-WO-normalization_{idx}.png", title="Mel Spectrogram")

            write(f"{cfg.train.log_dir}/audio_{idx}.wav", cfg.data.sampling_rate, audio_generated)


if __name__ == "__main__":
    hydra_main()
