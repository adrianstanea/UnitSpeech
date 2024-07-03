import argparse
import logging
import os

import torch

from conf.hydra_config import (
    MainConfig,
)
from unitspeech.duration_predictor import DurationPredictor
from unitspeech.encoder import Encoder
from unitspeech.text import cleaned_text_to_sequence, phonemize, symbols
from unitspeech.unitspeech import UnitSpeech
from unitspeech.util import (
    fix_len_compatibility,
    get_phonemizer,
    get_vocoder,
    intersperse,
    load_speaker_embs,
    save_plot,
)
from scipy.io.wavfile import write
from subprocess import PIPE, Popen

from conf.hydra_config import (
    MainConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference.py")
cfg = MainConfig


def main():
    global_phonemizer = get_phonemizer(cfg.inference.language)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")
    if device.type == "cpu" and cfg.train.on_GPU:
        raise ValueError("CUDA is not available.")
    logger.info(f"Running on: {device.type}")

    os.makedirs(cfg.train.log_dir, exist_ok=True)

    num_downsamplings_in_unet = len(cfg.decoder.dim_mults) - 1
    out_size = fix_len_compatibility(
        cfg.train.out_size_second * cfg.data.sampling_rate // cfg.data.hop_length,
        num_downsamplings_in_unet=num_downsamplings_in_unet,
    )

    # Load and initialize Vocoder
    vocoder = get_vocoder(config_path=cfg.vocoder.config_path, checkpoint=cfg.vocoder.ckpt_path, device=device)

    ## LOAD CHECKPOINTS FOR EACH MODULE
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
    if cfg.inference.use_finetuned_decoder:
        logger.info("Using finetuned decoder")
        decoder_dict = torch.load(
            f"{cfg.finetune.finetuned_decoders_path}/{cfg.inference.ID}.pt", map_location=lambda loc, storage: loc
        )
        decoder.load_state_dict(decoder_dict["model"])
    else:
        logger.info("Using generic decoder")
        decoder_dict = torch.load(cfg.decoder.checkpoint, map_location=lambda loc, storage: loc)
        decoder.load_state_dict(decoder_dict["model"])
    _ = decoder.eval()

    logger.info("Initializing the Text Encoder...")
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
    _ = duration_predictor.eval()

    logger.info("Loading spectrogram normalization params")
    mel_max = decoder_dict["mel_max"].to(device)
    mel_min = decoder_dict["mel_min"].to(device)

    text: str = cfg.inference.text
    SPKR_ID = torch.LongTensor([int(cfg.inference.ID)]).to(device)

    logger.info(f"Speaker ID: {SPKR_ID.item()}")
    logger.info(f"Text: {text}")
    # Process text
    logger.info("Processing text")
    phoneme = phonemize(text, global_phonemizer)
    logger.info(f"Phonemes: {phoneme}")
    phoneme = cleaned_text_to_sequence(phoneme)
    phoneme = intersperse(phoneme, len(symbols))
    phoneme = torch.LongTensor(phoneme).cuda().unsqueeze(0)
    phoneme_lengths = torch.LongTensor([phoneme.shape[-1]]).cuda()

    spk_emb = decoder_dict["spk_emb"].to(device)

    logger.info("Running inference")
    with torch.no_grad():
        y_enc, y_dec, attn = decoder.execute_text_to_speech(
            phoneme=phoneme,
            phoneme_lengths=phoneme_lengths,
            spk_emb=spk_emb,
            text_encoder=text_encoder,
            duration_predictor=duration_predictor,
            num_downsamplings_in_unet=num_downsamplings_in_unet,
            diffusion_steps=cfg.inference.diffusion_steps,
            length_scale=cfg.inference.length_scale,
            text_gradient_scale=cfg.inference.text_gradient_scale,
            spk_gradient_scale=cfg.inference.spk_gradient_scale,
        )
        mel_generated = (y_dec + 1) / 2 * (mel_max - mel_min) + mel_min
        audio_generated = vocoder.forward(mel_generated).cpu().squeeze().clamp(-1, 1).numpy()

    if cfg.inference.with_plot:
        save_plot(y_dec.squeeze().cpu(), f"{cfg.train.log_dir}/decoder-normalized.png", title="Mel Spectrogram")
        save_plot(attn.squeeze().cpu(), f"{cfg.train.log_dir}/attention.png", title="Attention")
        save_plot(y_enc.squeeze().cpu(), f"{cfg.train.log_dir}/encoder.png", title="Encoder")
        save_plot(
            mel_generated.squeeze().cpu(), f"{cfg.train.log_dir}/decoder-WO-normalization.png", title="Mel Spectrogram"
        )

    write(f"{cfg.train.log_dir}/{cfg.inference.file_path}", cfg.data.sampling_rate, audio_generated)

    if cfg.inference.with_sv56_normalization:
        cmd = "python3 sv56_inplace.py --in_dir {cfg.train.log_dir}"
        p = Popen(cmd, shell=True, stdout=PIPE)
        r = p.wait()
        if r != 0:
            raise RuntimeError("Audio gain normalization failed to execute.")

    os.system(f"cp {cfg.train.log_dir}/{cfg.inference.file_path} {os.getcwd()}/{cfg.inference.file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated_sample_path",
        type=str,
        default=cfg.inference.file_path,
        help="The path to save the generated audio.",
    )

    parser.add_argument('--text', type=str, required=True,
                        help='The desired transcript to be generated.')
    parser.add_argument(
        "--ID", type=int, default=cfg.inference.ID, help="The speaker ID to be used for the generation."
    )
    parser.add_argument(
        "--text_gradient_scale",
        type=float,
        default=cfg.inference.text_gradient_scale,
        help="Gradient scale of classifier-free guidance (cfg) for text condition. (0.0: wo cfg)",
    )
    parser.add_argument(
        "--spk_gradient_scale",
        type=float,
        default=cfg.inference.spk_gradient_scale,
        help="Gradient scale of classifier-free guidance (cfg) for speaker condition. (0.0: wo cfg)",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=cfg.inference.length_scale,
        help="The parameter for adjusting speech speed. The smaller it is compared to 1, the faster the speech becomes.",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=cfg.inference.diffusion_steps,
        help="The number of iterations for sampling in the diffusion model.",
    )
    args = parser.parse_args()

    # Reassign some values if necessary - hydra didnt support romanian characters from CLI
    cfg.inference.file_path = args.generated_sample_path
    cfg.inference.text = args.text
    cfg.inference.ID = args.ID
    cfg.inference.text_gradient_scale = args.text_gradient_scale
    cfg.inference.spk_gradient_scale = args.spk_gradient_scale
    cfg.inference.length_scale = args.length_scale
    cfg.inference.diffusion_steps = args.diffusion_steps

    main()
