import logging
import os

import hydra
import numpy as np
import torch
import torchaudio as ta
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, to_absolute_path
from tqdm import tqdm

from conf.hydra_config import TrainingUnitEncoderConfig
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN
from unitspeech.util import parse_filelist
from unitspeech.vocoder.meldataset import mel_spectrogram

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig)


def generate_mels(filelist_path, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    filepaths_and_text = parse_filelist(filelist_path, split_char='|')

    base_dir = os.path.join(os.path.dirname(filepaths_and_text[0][0]), "mels")
    os.makedirs(base_dir, exist_ok=True)

    logging.info(f"Generating mels for {len(filepaths_and_text)} files at {to_absolute_path(base_dir)}")
    for idx, (filepath, _) in tqdm(enumerate(filepaths_and_text),
                                   total=len(filepaths_and_text)):
        audio, sr = ta.load(filepath)
        audio = audio

        if sr != sampling_rate:
            logging.info(f"Resampling {filepath} from {sr} to {sampling_rate}")
            audio = ta.transforms.Resample(orig_freq=sr,
                                           new_freq=sampling_rate)(audio)

        mel = mel_spectrogram(audio,
                              n_fft=n_fft,
                              num_mels=num_mels,
                              sampling_rate=sampling_rate,
                              hop_size=hop_size,
                              win_size=win_size,
                              fmin=fmin,
                              fmax=fmax,
                              center=False)
        np.save(os.path.join(base_dir, os.path.basename(filepath).replace(".wav", ".npy")), mel)

def generate_spkr_embeddings(filelist_path, spkr_embedder, sample_rate, device):
    filepaths_and_text = parse_filelist(filelist_path, split_char='|')

    base_dir = os.path.join(os.path.dirname(filepaths_and_text[0][0]), "speaker_embeddings")
    os.makedirs(base_dir, exist_ok=True)


    logging.info(f"Generating speaker embeddings for {len(filepaths_and_text)} files at {to_absolute_path(base_dir)}")
    for idx, (filepath, _) in tqdm(enumerate(filepaths_and_text),
                                   total=len(filepaths_and_text)):
        audio, sr = ta.load(filepath)
        audio = audio.to(device)

        if sr != sample_rate:
            logging.info(f"Resampling {filepath} from {sr} to {sample_rate}")
            audio = ta.transforms.Resample(orig_freq=sr,
                                           new_freq=sample_rate).to(device)(audio)

        spkr_emb = spkr_embedder(audio).detach().cpu().numpy()
        np.save(os.path.join(base_dir, os.path.basename(filepath).replace(".wav", ".npy")), spkr_emb)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: TrainingUnitEncoderConfig):
    os.chdir(get_original_cwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spkr_embedder = ECAPA_TDNN(feat_dim=cfg.spkr_encoder.feat_dim,
                            channels=cfg.spkr_encoder.channels,
                            emb_dim=cfg.spkr_encoder.spk_emb_dim,
                            feat_type=cfg.spkr_encoder.feat_type,
                            sr=cfg.spkr_encoder.sr,
                            feature_selection=cfg.spkr_encoder.feature_selection,
                            update_extract=cfg.spkr_encoder.update_extract,
                            config_path=cfg.spkr_encoder.config_path).to(device)
    if not os.path.exists(cfg.spkr_encoder.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint for speaker embedding extractor not found: {cfg.spkr_encoder.checkpoint}")
    state_dict = torch.load(cfg.spkr_encoder.checkpoint,
                            map_location=lambda loc, storage: loc)
    spkr_embedder.load_state_dict(state_dict["model"], strict=False)
    _ = spkr_embedder.eval()  # FREEZING SPEAKER EMBEDDER
    for param in spkr_embedder.parameters():
        param.requires_grad = False

    mels_config = {
        'n_fft': cfg.data.n_fft,
        'num_mels': cfg.data.n_feats,
        'sampling_rate': cfg.data.sampling_rate,
        'hop_size': cfg.data.hop_length,
        'win_size': cfg.data.win_length,
        'fmin': cfg.data.mel_fmin,
        'fmax': cfg.data.mel_fmax,
    }
    generate_mels(filelist_path = cfg.data.train_filelist_path,
                  **mels_config)
    generate_mels(filelist_path = cfg.data.test_filelist_path,
                  **mels_config)

    generate_spkr_embeddings(filelist_path=cfg.data.train_filelist_path,
                             spkr_embedder=spkr_embedder,
                             sample_rate=cfg.data.sampling_rate,
                             device=device)

    generate_spkr_embeddings(filelist_path=cfg.data.test_filelist_path,
                             spkr_embedder=spkr_embedder,
                             sample_rate=cfg.data.sampling_rate,
                             device=device)



if __name__ == "__main__":
    main()