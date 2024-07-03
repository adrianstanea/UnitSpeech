import argparse
import logging
import os

import librosa
import torch
import torchaudio
from tqdm import tqdm

from conf.hydra_config import (
    MainConfig,
)
from unitspeech.encoder import Encoder
from unitspeech.unitspeech import UnitSpeech
from unitspeech.util import (
    fix_len_compatibility,
    generate_path,
    get_speaker_embedder,
    get_unit_extracter,
    process_unit,
    sequence_mask,
)
from unitspeech.vocoder.meldataset import mel_spectrogram

from conf.hydra_config import (
    MainConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune.py")
cfg = MainConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")
    if device.type == "cpu" and cfg.train.on_GPU:
        raise ValueError("CUDA is not available.")
    logger.info(f"Running on: {device.type}")

    num_downsamplings_in_unet = len(cfg.decoder.dim_mults) - 1
    out_size = fix_len_compatibility(
        cfg.train.out_size_second * cfg.data.sampling_rate // cfg.data.hop_length,
        num_downsamplings_in_unet=num_downsamplings_in_unet,
    )
    # ================================================================================
    print("Initializing Speaker Encoder...")
    spk_embedder = get_speaker_embedder(device, cfg.spkr_embedder)
    print("Initializing Unit Extracter...")
    unit_extractor = get_unit_extracter(device, cfg.unit_extractor)

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
    decoder_dict = torch.load(cfg.decoder.checkpoint, map_location=lambda loc, storage: loc)
    decoder.load_state_dict(decoder_dict["model"])
    _ = decoder.train()

    logger.info("Initializing unit encoder for finetunning...")
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
    unit_encoder_dict = torch.load(cfg.unit_encoder.checkpoint, map_location=lambda storage, loc: storage)
    unit_encoder.load_state_dict(unit_encoder_dict["model"])
    _ = unit_encoder.eval()

    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=cfg.finetune.learning_rate)
    if cfg.train.fp16_run:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("Loading spectrogram normalization params")
    wav, sr = librosa.load(cfg.finetune.reference_sample)
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
    ).to(device)
    mel_max = decoder_dict["mel_max"].to(device)
    mel_min = decoder_dict["mel_min"].to(device)
    # BAD TO USE REFERENCE AUDIO PARAMS
    # mel_min = mel.min(-1, keepdim=True)[0]
    # mel_max = mel.max(-1, keepdim=True)[0]
    mel = (mel - mel_min) / (mel_max - mel_min) * 2 - 1
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
    # =====================================================================================
    # Finetune the decoder
    for _ in tqdm(range(cfg.finetune.n_iters)):
        cond_x = cond_x.detach()
        mel = mel.detach()
        mel_mask = mel_mask.detach()
        mel_lengths = mel_lengths.detach()
        spk_emb = spk_emb.detach()
        attn = attn.detach()

        decoder.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfg.train.fp16_run):
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

        if cfg.train.fp16_run:
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
    logger.info("Saving the finetuned decoder...")
    os.makedirs(f"{cfg.finetune.finetuned_decoders_path}", exist_ok=True)
    decoder_dict["model"] = decoder.state_dict()
    decoder_dict["mel_min"] = mel_min
    decoder_dict["mel_max"] = mel_max
    decoder_dict["spk_emb"] = spk_emb
    torch.save(decoder_dict, f"{cfg.finetune.finetuned_decoders_path}/{cfg.finetune.ID}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reference audio
    parser.add_argument(
        "--reference_sample",
        type=str,
        default=cfg.finetune.reference_sample,
        help="Sample used to adapt the model to the speaker.",
    )
    parser.add_argument(
        "--ID", type=int, default=cfg.finetune.ID, help="Unique value used to identify the finetuned decoder."
    )
    parser.add_argument('--n_iters', type=int, default=500,
                    help='Number of fine-tuning iterations.')
    parser.add_argument('--learning_rate', type=int, default=2e-5,
                        help='Learning rate of the optimizer during fine-tuning.')

    args = parser.parse_args()
    # Reassign some values if necessary - hydra didnt support romanian characters from CLI
    cfg.finetune.reference_sample = args.reference_sample
    cfg.finetune.ID = args.ID
    cfg.finetune.learning_rate = args.learning_rate
    cfg.finetune.n_iters = args.n_iters

    main()
