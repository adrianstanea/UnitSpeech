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
from ast import Raise
from itertools import chain

import hydra
import monotonic_align
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from conf.hydra_config import (
    AdamConfig,
    LJSPeechConfig,
    TrainingUnitEncoderConfig_STEP1,
)
from data import TextMelSpeakerBatchCollate, TextMelSpeakerDataset
from unitspeech.duration_predictor import DurationPredictor
from unitspeech.encoder import Encoder
from unitspeech.unitspeech import UnitSpeech
from unitspeech.util import (
    duration_loss,
    fix_len_compatibility,
    load_speaker_embs,
    plot_tensor,
    save_plot,
    sequence_mask,
)

# from utils import create_symlink

logger = logging.getLogger("train_step1.py")
logger.setLevel(logging.DEBUG)

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig_STEP1)
cs.store(group="dataset", name="LJSpeech", node=LJSPeechConfig)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig_STEP1):
    # =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")
    if device.type == "cpu" and cfg.train.on_GPU:
        raise ValueError("CUDA is not available.")
    logger.info(f"Running on: {device.type}")
    # =============================================================================
    # Hydra: keep the original working directory but save logs in the hydra output directory
    output_dir = os.getcwd()  # Hydra changes the working directory when running the script
    os.chdir(get_original_cwd())
    cfg.train.log_dir = os.path.join(output_dir, cfg.train.log_dir)

    logger.debug(f"Hydra output dir located at: {output_dir}")
    logger.debug(f"Running from: {os.getcwd()}")
    logger.info(f"logging data into: {cfg.train.log_dir}")
    # =============================================================================
    # TODO: remap .pt save paths: on DGX should be available from data partition as mount
    # logger.info("Creating a symlink to dataset...")
    # local_dir = "DUMMY"
    # host_dir = os.path.join('/datasets', cfg.dataset.path)
    # create_symlink(local_dir, host_dir)
    # local_dir = "unitspeech/checkpoints"
    # host_dir = "/checkpoints"
    # create_symlink(local_dir, host_dir)
    # =============================================================================
    # TODO (OPTIONAL): HYPERPARAMTERES
    num_downsamplings_in_unet = len(cfg.decoder.dim_mults) - 1
    # out_size - a slice of mel-spectrogram to train the decoder on. Slice both text and mel-spectrogram.
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
    logger.info("Initializing datasets...")
    logger.info("Loading train dataset...")
    train_dataset = TextMelSpeakerDataset(filelist_path=cfg.dataset.train_filelist_path,
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
                                          mel_max_path=cfg.dataset.mel_max_path)
    batch_collate = TextMelSpeakerBatchCollate()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.train.batch_size,
                              collate_fn=batch_collate,
                              drop_last=cfg.train.drop_last,
                              num_workers=cfg.train.num_workers,
                              shuffle=cfg.train.shuffle)
    logger.info("Loading validation dataset...")
    test_dataset = TextMelSpeakerDataset(filelist_path=cfg.dataset.test_filelist_path,
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
                                        mel_max_path=cfg.dataset.mel_max_path)
    # =============================================================================
    logging.info("Loading speaker embeddings")
    pretrained_embs = load_speaker_embs(embs_path =os.path.join(cfg.data.embs_path, cfg.dataset.name),
                                        normalize=True)
    speaker_embeddings = torch.nn.Embedding.from_pretrained(pretrained_embs,
                                                            freeze=True).to(device=device)
    # =============================================================================
    logger.info("Initializing decoder: GradTTS ...")
    # UnitSpeech uses GradTTS as the diffusion-based decoder model
    decoder = UnitSpeech(n_feats=cfg.data.n_feats,
                         dim=cfg.decoder.dim,
                         dim_mults=cfg.decoder.dim_mults,
                         beta_min=cfg.decoder.beta_min,
                         beta_max=cfg.decoder.beta_max,
                         pe_scale=cfg.decoder.pe_scale,
                         spk_emb_dim=cfg.decoder.spk_emb_dim).to(device)
    logger.info(f"Number of parameters of the decoder: {decoder.nparams}")
    # =============================================================================
    logger.info("Initializing the Text Encoder...")
    text_encoder = Encoder(n_vocab=cfg.encoder.n_vocab,
                           n_feats=cfg.data.n_feats,
                           n_channels=cfg.encoder.n_channels,
                           filter_channels=cfg.encoder.filter_channels, # filter_channels_dp is missing in UnitSpeech
                           n_heads=cfg.encoder.n_heads,
                           n_layers=cfg.encoder.n_layers,
                           kernel_size=cfg.encoder.kernel_size,
                           p_dropout=cfg.encoder.p_dropout,
                           window_size=cfg.encoder.window_size).to(device)
    logger.info(f"Number of parameters of the text encoder: {text_encoder.nparams}")
    # =============================================================================
    logger.info("Initializing the Duration Predictor...")
    duration_predictor = DurationPredictor(in_channels=cfg.duration_predictor.in_channels,
                                           filter_channels=cfg.duration_predictor.filter_channels,
                                           kernel_size=cfg.duration_predictor.kernel_size,
                                           p_dropout=cfg.duration_predictor.p_dropout,
                                           spk_emb_dim=cfg.duration_predictor.spk_emb_dim).to(device)
    logger.info(f"Number of parameters of the duration predictor: {duration_predictor.nparams}")
    # =============================================================================
    logger.info("Initializing optimizer...")
    trainable_params = chain(text_encoder.parameters(),
                             duration_predictor.parameters(),
                             decoder.parameters())
    if OmegaConf.get_type(cfg.optimizer) is AdamConfig:
        optimizer = torch.optim.Adam(params=trainable_params,
                                     lr=cfg.optimizer.learning_rate)
    else:
        raise NotImplementedError("Only Adam optimizer is supported")
    # =============================================================================
    if cfg.train.fp16_run:
        scaler = torch.cuda.amp.GradScaler()
    # =============================================================================
    logger.info("Logging test batch...")
    test_batch = test_dataset.sample_test_batch(size=cfg.train.test_size)
    for item in test_batch:
        mel, spk = item['y'], item['spk_id']
        i = int(spk.cpu())
        writer.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{cfg.train.log_dir}/original_{i}.png')
    # =============================================================================
    logger.info(f"Training for {cfg.train.n_epochs} epochs...")
    iteration = 0
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(1, cfg.train.n_epochs + 1):
            text_encoder.train()
            duration_predictor.train()
            decoder.train()

            dur_losses, prior_losses, diff_losses = [], [], []
            with tqdm(train_loader, total=len(train_dataset)//cfg.train.batch_size) as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    if batch_idx >= 5:
                        continue
                    text_encoder.zero_grad()
                    duration_predictor.zero_grad()
                    decoder.zero_grad()

                    with torch.cuda.amp.autocast(enabled=cfg.train.fp16_run):
                        dur_loss, prior_loss, diff_loss = compute_train_step_loss(cfg,
                                                                                batch,
                                                                                speaker_embeddings,
                                                                                text_encoder,
                                                                                duration_predictor,
                                                                                decoder,
                                                                                out_size)
                    loss = sum([dur_loss, prior_loss, diff_loss])

                    if cfg.train.fp16_run:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        text_encoder_grad_norm = torch.nn.utils.clip_grad_norm_(text_encoder.parameters(),
                                                                                max_norm=1)
                        duration_predictor_grad_norm = torch.nn.utils.clip_grad_norm_(duration_predictor.parameters(),
                                                                                    max_norm=1)
                        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                                                        max_norm=1)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        text_encoder_grad_norm = torch.nn.utils.clip_grad_norm_(text_encoder.parameters(),
                                                                                max_norm=1)
                        duration_predictor_grad_norm = torch.nn.utils.clip_grad_norm_(duration_predictor.parameters(),
                                                                                    max_norm=1)
                        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(),
                                                                        max_norm=1)
                        optimizer.step()

                    writer.add_scalar("Train1/duration_loss",
                                    dur_loss.item(), global_step=iteration)
                    writer.add_scalar("Train1/prior_loss",
                                    prior_loss.item(), global_step=iteration)
                    writer.add_scalar("Train1/diff_loss",
                                    diff_loss.item(), global_step=iteration)
                    writer.add_scalar("Train1/text_encoder_grad_norm",
                                    text_encoder_grad_norm, global_step=iteration)
                    writer.add_scalar("Train1/duration_predictor_grad_norm",
                                    duration_predictor_grad_norm, global_step=iteration)
                    writer.add_scalar("Train1/decoder_grad_norm",
                                    decoder_grad_norm, global_step=iteration)

                    dur_losses.append(dur_loss.item())
                    prior_losses.append(prior_loss.item())
                    diff_losses.append(diff_loss.item())

                    # Update progress bar every X batches
                    if batch_idx % 5 == 0:
                        msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                        progress_bar.set_description(msg)
                    iteration += 1

            log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
            log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
            log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
            with open(f'{cfg.train.log_dir}/train.log', 'a') as f:
                f.write(log_msg)

            # Evaluate and save model checkpoint every X epochs
            if epoch % cfg.train.save_every > 0:
                continue

            logger.info('Synthesis...')
            text_encoder.eval()
            duration_predictor.eval()
            decoder.eval()
            # with torch.no_grad():
            #     for i, item in enumerate(test_batch):
            #         # batch = 1 => we iterate sample by sample
            #         x = item['x'].to(torch.long).unsqueeze(0).cuda() # phonemes: (batch, phonemes)
            #         x_lengths = torch.LongTensor([x.shape[-1]]).cuda() # phoneme lengths: (batch)
            #         spk_id = item['spk_id'].cuda() # (batch)
            #         spk_embs = speaker_embeddings(spk_id).unsqueeze(1).cuda()


            #         y_enc, y_dec, attn = decoder.execute_text_to_speech(phoneme = x,
            #                                                             phoneme_lengths=x_lengths,
            #                                                             spk_emb=spk_embs,
            #                                                             text_encoder=text_encoder,
            #                                                             duration_predictor=duration_predictor,
            #                                                             num_downsamplings_in_unet=num_downsamplings_in_unet,
            #                                                             diffusion_steps=cfg.decoder.diffusion_steps)
            #         # Scale back the generated mel = y_dec
            #         y_dec = ((y_dec + 1) / 2 * (train_dataset.mel_max.to(y_dec.device) - train_dataset.mel_min.to(y_dec.device)))
            #         writer.add_image(f'image_{i}/generated_enc',
            #                         plot_tensor(y_enc.squeeze().cpu()),
            #                         global_step=iteration, dataformats='HWC')
            #         writer.add_image(f'image_{i}/generated_dec',
            #                         plot_tensor(y_dec.squeeze().cpu()),
            #                         global_step=iteration, dataformats='HWC')
            #         writer.add_image(f'image_{i}/alignment',
            #                         plot_tensor(attn.squeeze().cpu()),
            #                         global_step=iteration, dataformats='HWC')
            #         save_plot(y_enc.squeeze().cpu(),
            #                 f'{cfg.train.log_dir}/generated_enc_{i}.png')
            #         save_plot(y_dec.squeeze().cpu(),
            #                 f'{cfg.train.log_dir}/generated_dec_{i}.png')
            #         save_plot(attn.squeeze().cpu(),
            #                 f'{cfg.train.log_dir}/alignment_{i}.png')

            logger.info(f"Saving checkpoints at epoch {epoch}...")
            os.makedirs(f"{cfg.train.log_dir}/checkpoints_{epoch}", exist_ok=True)

            text_encoder_ckpt = {"model": text_encoder.state_dict(),
                                "spk_emb": speaker_embeddings.state_dict(),
                                "mel_min": train_dataset.mel_min,
                                "mel_max": train_dataset.mel_max}
            logger.debug("Saving text encoder checkpoint")
            torch.save(text_encoder_ckpt,
                    f"{cfg.train.log_dir}/checkpoints_{epoch}/text_encoder.pt")

            duration_predictor_ckpt = {"model": duration_predictor.state_dict()}
            logger.debug("Saving duration predictor checkpoint")
            torch.save(duration_predictor_ckpt,
                    f"{cfg.train.log_dir}/checkpoints_{epoch}/duration_predictor.pt")

            pretrained_decoder_ckpt = {"model": decoder.state_dict()}
            logger.debug("Saving decoder checkpoint")
            torch.save(pretrained_decoder_ckpt,
                    f"{cfg.train.log_dir}/checkpoints_{epoch}/pretrained_decoder.pt")


def compute_train_step_loss(cfg: TrainingUnitEncoderConfig_STEP1,
                            batch,
                            pretained_spk_embs,
                            text_encoder: Encoder,
                            duration_predictor: DurationPredictor,
                            decoder: UnitSpeech,
                            out_size):
    x_sample, x_sample_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()  # TEXT
    y_sample, y_sample_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()  # MEL
    spk_id = batch['spk_id'].cuda()
    spk_embs = pretained_spk_embs(spk_id).unsqueeze(1).cuda()
    # SPEAKER EMBEDDING
    # spk_emb_sample = batch['spk_emb'].cuda() #ORIGINAL
    # spk_emb_sample = batch['spk_id'].cuda().unsqueeze(1) # Adapt for duration predictor
    # spk_embs = load_speaker_emb()

    mu_x, x, x_mask = text_encoder(x_sample, x_sample_lengths)
    logw = duration_predictor(x, x_mask, w=None, g=spk_embs, reverse=True)
    y_max_length = y_sample.shape[-1]

    y_mask = sequence_mask(y_sample_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) # (batch, 1, x_max_length, y_max_length)

    # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
    with torch.no_grad():
        const = -0.5 * math.log(2 * math.pi) * cfg.data.n_feats
        factor = -0.5 * torch.ones(mu_x.shape,dtype=mu_x.dtype, device=mu_x.device)
        y_square = torch.matmul(factor.transpose(1, 2), y_sample ** 2)
        y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y_sample)
        mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
        log_prior = y_square - y_mu_double + mu_square + const

        attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
        attn = attn.detach()

    # Compute loss between predicted log-scaled durations and those obtained from MAS
    logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
    dur_loss = duration_loss(logw, logw_, x_sample_lengths)

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
        y_cut_lengths = torch.LongTensor(y_cut_lengths) # Type Error in the IDE but works at runtime
        y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)

        attn = attn_cut # (batch, max_phoneme_length, mel_cut)
        y = y_cut # (batch, mels, mel_cut)
        y_mask = y_cut_mask

    # Align encoded text with mel-spectrogram and get mu_y segment
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2).contiguous(), mu_x.transpose(1, 2).contiguous())
    mu_y = mu_y.transpose(1, 2).contiguous() # (batch, mels, mel_cut)
    # TODO: review this step: mask is not applied here in the original GradTTS implementation. Instead decoder uses it in the loss fn
    mu_y = mu_y * y_mask

    # Compute loss of score-based decoder
    # y: (batch, mels, cut_of_original_audio)
    # y_mask 
    # mu_y - features 
    diff_loss, xt = decoder.compute_loss(y, y_mask, mu_y, spk_emb=spk_embs)

    # Compute loss between aligned encoder outputs and mel-spectrogram => Loss encoder
    prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
    prior_loss = prior_loss / (torch.sum(y_mask) * cfg.data.n_feats)

    return dur_loss, prior_loss, diff_loss


if __name__ == "__main__":
    hydra_main()
