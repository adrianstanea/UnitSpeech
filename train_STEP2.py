"""
First step of training UnitSpeech: Train the Diffusion-Based Text-to-Speech Model

Components:
    - Text Encoder: Encodes phoneme embeddings into a sequence of hidden states.
    - Duration Predictor: Predicts the duration of each phoneme to produce same length as mel-spectrogram.
    - Decoder: Diffusion-based model that generates mel-spectrogram from hidden states.
"""
import logging
import math
import os
import random
from itertools import chain

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from conf.hydra_config import AdamConfig, TrainingUnitEncoderConfig_STEP2
from data import UnitDurationMelSPeakerDataset, UnitDurationMelSpeakerBatchCollate
from unitspeech.duration_predictor import DurationPredictor
from unitspeech.encoder import Encoder
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.unitspeech import UnitSpeech
from unitspeech.util import (
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)
from utils import create_symlink

logger = logging.getLogger("train_step1.py")
logger.setLevel(logging.DEBUG)

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig_STEP2)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig_STEP2):
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.cuda_device else "cpu")
    if device.type == "cpu" and not cfg.train.cuda_device:
        raise ValueError("CUDA is not available.")
    logger.info(f"Running on: {device.type}")
    # =============================================================================
    # Hydra: keep the original working directory but save logs in the hydra output directory
    output_dir = os.getcwd() # Hydra changes the working directory when running the script
    os.chdir(get_original_cwd())
    logger.debug(f"Hydra output dir located at: {output_dir}")
    logger.debug(f"Running from: {os.getcwd()}")

    cfg.train.log_dir = output_dir + '/' + cfg.train.log_dir
    logger.info(f"logging data into: {cfg.train.log_dir}")
    # =============================================================================
    # TODO: remap .pt save paths: on DGX should be available from data partition as mount
    logger.info("Creating a symlink to dataset...")
    local_dir = "DUMMY"
    host_dir = os.path.join('/datasets', cfg.dataset.path)
    create_symlink(local_dir, host_dir)
    local_dir = "unitspeech/checkpoints"
    host_dir = "/checkpoints"
    create_symlink(local_dir, host_dir)
    # =============================================================================
    # TODO (OPTIONAL): HYPERPARAMTERES
    num_downsamplings_in_unet = len(cfg.decoder.dim_mults) - 1
    out_size = fix_len_compatibility(cfg.train.out_size_second * cfg.data.sampling_rate // cfg.data.hop_length,
                                    num_downsamplings_in_unet=num_downsamplings_in_unet)
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
    # logger.info("Initializing the Speaker Encoder... ")
    # spkr_encoder = ECAPA_TDNN(feat_dim=cfg.spkr_encoder.feat_dim,
    #                           channels=cfg.spkr_encoder.channels,
    #                           emb_dim=cfg.spkr_encoder.spk_emb_dim,
    #                           feat_type=cfg.spkr_encoder.feat_type,
    #                           sr=cfg.spkr_encoder.sr,
    #                           feature_selection=cfg.spkr_encoder.feature_selection,
    #                           update_extract=cfg.spkr_encoder.update_extract,
    #                           config_path=cfg.spkr_encoder.config_path).to(device)
    # if not os.path.exists(cfg.spkr_encoder.checkpoint):
    #     raise FileNotFoundError(f"Checkpoint for speaker embedding extractor not found: {cfg.spkr_encoder.checkpoint}")
    # state_dict = torch.load(cfg.spkr_encoder.checkpoint,
    #                         map_location=lambda loc, storage: loc)
    # spkr_encoder.load_state_dict(state_dict["model"], strict=False)
    # # _ = spkr_encoder.to(device).eval()  # FREEZING SPEAKER EMBEDDER
    # _ = spkr_encoder.eval()  # FREEZING SPEAKER EMBEDDER
    # for param in spkr_encoder.parameters():
    #     param.requires_grad = False
    # =============================================================================
    logger.info("Initializing model (GradTTS checkpoint)...")
    # UnitSpeech uses GradTTS as the decoder model: pretrain decoder prior to training the unit encoder
    decoder = UnitSpeech(n_feats=cfg.data.n_feats,
                        dim=cfg.decoder.dim,
                        dim_mults=cfg.decoder.dim_mults,
                        beta_min=cfg.decoder.beta_min,
                        beta_max=cfg.decoder.beta_max,
                        pe_scale=cfg.decoder.pe_scale,
                        spk_emb_dim=cfg.decoder.spk_emb_dim).to(device)
    logger.info(f"Number of parameters of the decoder: {decoder.nparams}")
    if not os.path.exists(cfg.decoder.checkpoint):
        raise FileNotFoundError(f"Checkpoint for decoder not found: {cfg.decoder.checkpoint}")
    decoder_dict = torch.load(cfg.decoder.checkpoint,
                            map_location=lambda loc,
                            storage: loc)
    decoder.load_state_dict(decoder_dict["model"])
    logger.info(f"Loaded GradTTS checkpoint from {cfg.decoder.checkpoint}")
    # TODO: how do i need to set the decoder to train the Unit Encoder ??  What kind of loss does it need ??
    # NOTE: might need to leave the decoder in train mode but with frozen params: 
    # Freeze the DDPM decoder
    _ = decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    # =============================================================================
    logger.info("Initializing UnitExtractor...")
    unit_extractor = SpeechEncoder.by_name(dense_model_name=cfg.unit_extractor.dense_model_name,
                                           quantizer_model_name=cfg.unit_extractor.quantizer_name,
                                           vocab_size=cfg.unit_extractor.vocab_size,
                                           deduplicate=cfg.unit_extractor.deduplicate,
                                           need_f0=cfg.unit_extractor.need_f0).to(device)
    # Freeze the unit extractor
    _ = unit_extractor.eval()
    for param in unit_extractor.parameters():
        param.requires_grad = False
    # =============================================================================
    logger.info("Initializing datasets...")
    # TODO: see what kind of features we need to extract when training the Unit Encoder => adapt dataset features
    train_dataset = UnitDurationMelSPeakerDataset(filelist_path=cfg.dataset.train_filelist_path,
                                                unit_extractor=unit_extractor,
                                                random_seed=cfg.train.seed,
                                                add_blank=cfg.data.add_blank,
                                                n_fft=cfg.data.n_fft,
                                                n_mels=cfg.data.n_feats,
                                                sample_rate=cfg.data.sampling_rate,
                                                hop_length=cfg.data.hop_length,
                                                win_length=cfg.data.win_length,
                                                f_min=cfg.data.mel_fmin,
                                                f_max=cfg.data.mel_fmax,
                                                load_preprocessed=True)
    batch_collate = UnitDurationMelSpeakerBatchCollate()
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=cfg.train.batch_size,
                            collate_fn=batch_collate,
                            drop_last=cfg.train.drop_last,
                            num_workers=cfg.train.num_workers,
                            shuffle=cfg.train.shuffle)
    # =============================================================================
    logger.info("Initializing the Unit Encoder...")
    unit_encoder = Encoder(n_vocab=cfg.data.n_units,
                        n_feats=cfg.data.n_feats,
                        n_channels=cfg.encoder.n_channels,
                        filter_channels=cfg.encoder.filter_channels,
                        n_heads=cfg.encoder.n_heads,
                        n_layers=cfg.encoder.n_layers,
                        kernel_size=cfg.encoder.kernel_size,
                        p_dropout=cfg.encoder.p_dropout,
                        window_size=cfg.encoder.window_size).to(device)
    logging.info(f"Number of parameters of the unit encoder: {unit_encoder.nparams}")
    # =============================================================================
    # We DONT USE IT HERE -> TAKE THE DURATIONS FROM THE UNIT EXTRACTOR
    # logger.info("Initializing the Duration Predictor...")
    # # TODO: is it needed on step 2?
    # duration_predictor = DurationPredictor(in_channels=cfg.duration_predictor.in_channels,
    #                                        filter_channels=cfg.duration_predictor.filter_channels,
    #                                        kernel_size=cfg.duration_predictor.kernel_size,
    #                                        p_dropout=cfg.duration_predictor.p_dropout,
    #                                        spk_emb_dim=cfg.duration_predictor.spk_emb_dim).to(device)
    # logger.info(f"Number of parameters of the duration predictor: {duration_predictor.nparams}")
    # =============================================================================
    # NOTE: optimizer should only update the unit encoder model, rest are frozen
    logger.info("Initializing optimizer...")
    # L_enc is the prior loss, L_grad is the diffusion loss
    trainable_params = chain(unit_encoder.parameters(),
                             decoder.parameters()) # We compute diffusion (L_grad) and encoder (L_enc) loss
    if OmegaConf.get_type(cfg.optimizer) is AdamConfig:
        optimizer = torch.optim.Adam(params=trainable_params,
                                     lr=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError("Only Adam optimizer is supported")
    # =============================================================================
    if cfg.train.fp16_run:
        scaler = torch.cuda.amp.GradScaler()

    # logger.info("Logging test batch...")
    # test_batch = test_dataset.sample_test_batch(size=cfg.train.test_size)
    # for i, item in enumerate(test_batch):
    #     # TODO: do we need te speaker id??
    #     mel, spk_emb = item['y'], item['spk_emb']
    #     writer.add_image(f'image_{i}/ground_truth',
    #                      plot_tensor(mel.squeeze()),
    #                      global_step=0,
    #                      dataformats='HWC')
    #     save_plot(mel.squeeze(),
    #               f'{cfg.train.log_dir}/original_{i}.png')

    logger.info(f"Training for {cfg.train.n_epochs} epochs...")
    iteration = 0
    for epoch in range(1, cfg.train.n_epochs + 1):
        unit_encoder.train()
        decoder.train() # TODO: review how it works when frozen gradients

        prior_losses, diff_losses = [], []
        with tqdm(train_loader, total=len(train_dataset)//cfg.train.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                unit_encoder.zero_grad()
                decoder.zero_grad()

                with torch.cuda.amp.autocast(enabled=cfg.train.fp16_run):
                    prior_loss, diff_loss = compute_train_step_loss(cfg,
                                                                    batch,
                                                                    unit_encoder,
                                                                    decoder,
                                                                    out_size)
                loss = sum([prior_loss, diff_loss])

                if cfg.train.fp16_run:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    text_encoder_grad_norm = torch.nn.utils.clip_grad_norm_(unit_encoder.parameters(), max_norm=1)
                    decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    text_encoder_grad_norm = torch.nn.utils.clip_grad_norm_(unit_encoder.parameters(), max_norm=1)
                    decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
                    optimizer.step()

                writer.add_scalar("Train1/prior_loss", prior_loss.item(), global_step=iteration)
                writer.add_scalar("Train1/diff_loss", diff_loss.item(), global_step=iteration)
                writer.add_scalar("Train1/text_encoder_grad_norm", text_encoder_grad_norm, global_step=iteration)
                writer.add_scalar("Train1/decoder_grad_norm", decoder_grad_norm, global_step=iteration)

                # dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                # Update progress bar every X batches
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    progress_bar.set_description(msg)
                iteration += 1

        log_msg = 'Epoch %d: ' % (epoch)
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{cfg.train.log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        # Evaluate and save model checkpoint every X epochs
        if epoch % cfg.train.save_every > 0:
            continue

        logger.info('Synthesis...')
        unit_encoder.eval()
        decoder.eval()


        logger.info(f"Saving checkpoints at epoch {epoch}...")
        os.makedirs(f"{cfg.train.log_dir}/checkpoints_{epoch}",exist_ok=True)

        unit_encoder_ckpt = {"model": unit_encoder.state_dict()}
        logger.debug("Saving unit encoder checkpoint")
        torch.save(unit_encoder_ckpt ,
                   f"{cfg.train.log_dir}/checkpoints_{epoch}/unit_encoder.pt")

def compute_train_step_loss(cfg: TrainingUnitEncoderConfig_STEP2,
                            batch,
                            unit_encoder : Encoder,
                            decoder : UnitSpeech,
                            out_size : int):
    """
    Computes 3 losses:
        1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
        2. prior loss: loss between mel-spectrogram and encoder outputs.
        3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

    Args:
        x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
        x_lengths (torch.Tensor): lengths of texts in batch.
        y (torch.Tensor): batch of corresponding mel-spectrograms.
        y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
        out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
            Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
    """
    x_unit, x_unit_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
    x_duration, x_duration_lengths = batch['x_duration'].cuda(), batch['x_duration_lengths'].cuda()
    y_sample, y_sample_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()  # MEL
    spk_emb_sample = batch['spk_emb'].cuda().unsqueeze(1)

    mu_x, x, x_mask = unit_encoder(x_unit, x_unit_lengths)

    # logw = duration_predictor(x, x_mask, w=None, g=spk_emb_sample, reverse=True) # The output of the duration predictor is the same as the durations from the unit extractor
    duration = x_duration



    y_max_length = y_sample.shape[-1]

    y_mask = sequence_mask(y_sample_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) # (batch, 1, x_max_length, y_max_length)
    attn = generate_path(duration,  attn_mask.squeeze(1)) # Use duration of unit extract rather than MAS

    # # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
    # with torch.no_grad():
    #     const = -0.5 * math.log(2 * math.pi) * cfg.data.n_feats
    #     factor = -0.5 * torch.ones(mu_x.shape,dtype=mu_x.dtype, device=mu_x.device)
    #     y_square = torch.matmul(factor.transpose(1, 2), y_sample ** 2)
    #     y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y_sample)
    #     mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
    #     log_prior = y_square - y_mu_double + mu_square + const

    #     attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
    #     attn = attn.detach()

    # NOTE: no duration loss is to be computet at this stage, only encoder and diffusion loss
    # Compute loss between predicted log-scaled durations and those obtained from MAS
    # logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
    # dur_loss = duration_loss(logw, logw_, x_sample_lengths)

    # Cut a small segment of mel-spectrogram in order to increase batch size
    if not isinstance(out_size, type(None)) and out_size < y_max_length:
        max_offset = (y_sample_lengths - out_size).clamp(0)
        offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
        out_offset = torch.LongTensor([torch.tensor(random.choice(range(start, end))
                                        if end > start else 0)
                                        for start, end in offset_ranges
        ]).to(y_sample_lengths)

        attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
        y_cut = torch.zeros(y_sample.shape[0], cfg.data.n_feats, out_size, dtype=y_sample.dtype, device=y_sample.device)
        y_cut_lengths = []
        for i, (y_, out_offset_) in enumerate(zip(y_sample, out_offset)):
            y_cut_length = out_size + (y_sample_lengths[i] - out_size).clamp(None, 0)
            y_cut_lengths.append(y_cut_length)
            cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
            y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
            attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
        y_cut_lengths = torch.LongTensor(y_cut_lengths)
        y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)

        # TODO: this is present in UnitSpeech but not in GradTTS
        # if y_cut_mask.shape[-1] < out_size:
        #     y_cut_mask = torch.nn.functional.pad(y_cut_mask, (0, out_size - y_cut_mask.shape[-1]))

        attn = attn_cut # (batch, max_phoneme_length, mel_cut)
        y = y_cut # (batch, mels, mel_cut)
        y_mask = y_cut_mask

    # Align encoded text with mel-spectrogram and get mu_y segment
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2).contiguous(),
                        mu_x.transpose(1, 2).contiguous())
    mu_y = mu_y.transpose(1, 2).contiguous() # (batch, mels, mel_cut)
    # cond_y = cond_y * y_mask # GradTTS skips this step

    # Compute loss of score-based decoder
    # y: (batch, mels, cut_of_original_audio)
    # y_mask 
    # mu_y - features 
    diff_loss, xt = decoder.compute_loss(y, y_mask, mu_y, spk_emb=spk_emb_sample)

    # Compute loss between aligned encoder outputs and mel-spectrogram => Loss encoder
    prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
    prior_loss = prior_loss / (torch.sum(y_mask) * cfg.data.n_feats)

    return prior_loss, diff_loss


if __name__ == "__main__":
    hydra_main()
