import argparse
import json
import logging
from math import log
import hydra
import librosa
import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

import torchaudio
from tqdm import tqdm

import data
from data.data import UnitMelSpeakerBatchCollate, UnitMelSpeakerDataset
from unitspeech.unitspeech import UnitSpeech
from unitspeech.encoder import Encoder
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import HParams, fix_len_compatibility, process_unit, generate_path, sequence_mask
from unitspeech.vocoder.env import AttrDict
from unitspeech.vocoder.meldataset import mel_spectrogram
from unitspeech.vocoder.models import BigVGAN
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN

from conf.hydra_config import AdamConfig, TrainingUnitEncoderConfig
from hydra.utils import get_original_cwd, to_absolute_path

from torch.utils.tensorboard import SummaryWriter


from hydra.core.config_store import ConfigStore

from utils import create_symlink


logger = logging.getLogger("train_unit-encoder.py")
logger.setLevel(logging.INFO)

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig):
    # =============================================================================
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info("CUDA is available.")
    else:
        raise ValueError("CUDA is not available.")
    # =============================================================================
    # Hydra: keep the original working directory but save logs in the hydra output directory
    os.chdir(get_original_cwd())
    logger.debug(f"Running from: {os.getcwd()}")
    logger.info(f"logging data into: {cfg.train.log_dir}")
    # =============================================================================
    # TODO: remap .pt save paths: on DGX should be available from data partition as mount
    # logger.info("Creating a symlink to dataset...")
    # source_dir_name = "DUMMY"
    # target_path = os.path.join('/datasets', 'LJSpeech')
    # create_symlink(source_dir_name, target_path)
    # =============================================================================
    # TODO (OPTIONAL): extra hyperparams
    # =============================================================================
    logger.info("Initializing random seed...")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    # =============================================================================
    logger.info("Initializing tensorboard...")
    writer = SummaryWriter(log_dir=to_absolute_path(cfg.train.log_dir))
    # =============================================================================
    logger.info("Initializing model (GradTTS checkpoint)...")
    # UnitSpeech uses GradTTS as the decoder model: pretrain decoder prior to training the unit encoder
    model = UnitSpeech(n_feats=cfg.data.n_feats,
                       dim=cfg.decoder.dim,
                       dim_mults=cfg.decoder.dim_mults,
                       beta_min=cfg.decoder.beta_min,
                       beta_max=cfg.decoder.beta_max,
                       pe_scale=cfg.decoder.pe_scale,
                       spk_emb_dim=cfg.decoder.spk_emb_dim)
    if not os.path.exists(cfg.decoder.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint for GradTTS decoder not found: {cfg.decoder.checkpoint}")
    state_dict = torch.load(cfg.decoder.checkpoint,
                            map_location=lambda loc, storage: loc)
    model.load_state_dict(state_dict["model"])
    _ = model.cuda().eval()  # FREEZING DECODER
    logger.info(f"Number of parameters of the model: {model.nparams}")
    # =============================================================================
    logger.info("Initializing the Speaker Encoder... ")
    spkr_encoder = ECAPA_TDNN(feat_dim=cfg.spkr_encoder.feat_dim,
                              channels=cfg.spkr_encoder.channels,
                              emb_dim=cfg.spkr_encoder.spk_emb_dim,
                              feat_type=cfg.spkr_encoder.feat_type,
                              sr=cfg.spkr_encoder.sr,
                              feature_selection=cfg.spkr_encoder.feature_selection,
                              update_extract=cfg.spkr_encoder.update_extract,
                              config_path=cfg.spkr_encoder.config_path)
    if not os.path.exists(cfg.spkr_encoder.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint for speaker embedding extractor not found: {cfg.spkr_encoder.checkpoint}")
    state_dict = torch.load(cfg.spkr_encoder.checkpoint,
                            map_location=lambda loc, storage: loc)
    spkr_encoder.load_state_dict(state_dict["model"], strict=False)
    _ = spkr_encoder.cuda().eval()  # FREEZING SPEAKER EMBEDDER
    # =============================================================================
    logger.info("Initializing UnitExtractor...")
    unit_extractor = SpeechEncoder.by_name(dense_model_name=cfg.unit_extractor.dense_model_name,
                                           quantizer_model_name=cfg.unit_extractor.quantizer_name,
                                           vocab_size=cfg.unit_extractor.vocab_size,
                                           deduplicate=cfg.unit_extractor.deduplicate,
                                           need_f0=cfg.unit_extractor.need_f0)
    _ = unit_extractor.cuda().eval()
    # =============================================================================
    unit_encoder = Encoder(n_vocab=cfg.data.n_units,
                           n_feats=cfg.data.n_feats,
                           n_channels=cfg.encoder.n_channels,
                           filter_channels=cfg.encoder.filter_channels,
                           n_heads=cfg.encoder.n_heads,
                           n_layers=cfg.encoder.n_layers,
                           kernel_size=cfg.encoder.kernel_size,
                           p_dropout=cfg.encoder.p_dropout,
                           window_size=cfg.encoder.window_size).cuda()
    logging.info(
        f"Number of parameters of the unit encoder: {unit_encoder.nparams}")
    # =============================================================================
    # =============================================================================
    # TODO: datasets => adapt for multispeaker
    logger.info("Initializing datasets...")
    train_dataset = UnitMelSpeakerDataset(filelist_path=cfg.data.train_filelist_path,
                                          unit_extractor=unit_extractor,
                                          n_fft=cfg.data.n_fft,
                                          n_mels=cfg.data.n_feats,
                                          sample_rate=cfg.data.sampling_rate,
                                          hop_length=cfg.data.hop_length,
                                          win_length=cfg.data.win_length,
                                          f_min=cfg.data.mel_fmin,
                                          f_max=cfg.data.mel_fmax,
                                          random_seed=cfg.train.seed)
    batch_collate = UnitMelSpeakerBatchCollate()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.train.batch_size,
                              collate_fn=batch_collate,
                              drop_last=cfg.train.drop_last,
                              num_workers=cfg.train.num_workers,
                              shuffle=cfg.train.shuffle)
    # =============================================================================
    # NOTE: optimizer should only update the unit encoder model, rest are frozen
    logger.info("Initializing optimizer...")
    if OmegaConf.get_type(cfg.optimizer) is AdamConfig:
        optimizer = torch.optim.Adam(params=unit_encoder.parameters(),
                                     lr=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError("Only Adam optimizer is supported")
    # =============================================================================

    if cfg.train.fp16_run:
        scaler = torch.cuda.amp.GradScaler()

    logger.info(f"Training for {cfg.train.n_epochs} epochs...")

    iteration = 0
    for epoch in range(1, cfg.train.n_epochs + 1):
        model.train()
        with tqdm(train_loader, total=len(train_dataset)//cfg.train.batch_size) as progress_bar:
           
           if cfg.train.fp16_run:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                _ = torch.nn.utils.clip_grad_norm_(unit_encoder.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
           else:
               loss.backward()
               _ = torch.nn.utils.clip_grad_norm_(unit_encoder.parameters(), max_norm=1)
               optimizer.step()




        # Save model checkpoint
        if epoch % cfg.train.save_every > 0:
            continue
        logger.info(f"Saving unit_encoder checkpoint at epoch {epoch}...")
        ckpt = model.state_dict()

    # =============================================================================


if __name__ == "__main__":
    hydra_main()
