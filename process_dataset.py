import logging
import os

import hydra
import numpy as np
import torch
import torchaudio as ta
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, to_absolute_path
from regex import D
from tqdm import tqdm

from conf.hydra_config import TrainingUnitEncoderConfig_STEP1
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import parse_filelist, process_unit
from unitspeech.vocoder.meldataset import mel_spectrogram

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig_STEP1)


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
            assert sampling_rate == 22_050, "Sampling rate should be 22_050"
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
            assert sample_rate == 16_000, "Sample rate should be 16_000"
            audio = ta.transforms.Resample(orig_freq=sr,
                                           new_freq=sample_rate).to(device)(audio)

        spkr_emb = spkr_embedder(audio)
        spkr_emb = spkr_emb / spkr_emb.norm() # They are used normalized in the other sample scripts
        np.save(os.path.join(base_dir, os.path.basename(filepath).replace(".wav", ".npy")), spkr_emb.detach().cpu().numpy())

def generate_unit_duration_embeddings(filelist_path, unit_extractor, sample_rate, hop_size, device):
    filepaths_and_text = parse_filelist(filelist_path, split_char='|')

    base_units = os.path.join(os.path.dirname(filepaths_and_text[0][0]), "units")
    os.makedirs(base_units, exist_ok=True)

    base_durations = os.path.join(os.path.dirname(filepaths_and_text[0][0]), "durations")
    os.makedirs(base_durations, exist_ok=True)

    logging.info(f"Generating units for {len(filepaths_and_text)} files at {to_absolute_path(base_units)}")
    logging.info(f"Generating durations for {len(filepaths_and_text)} files at {to_absolute_path(base_durations)}")

    for idx, (filepath, _) in tqdm(enumerate(filepaths_and_text),
                                   total=len(filepaths_and_text)):
        audio, sr = ta.load(filepath)
        audio = audio.to(device)

        if sr != sample_rate:
            assert sample_rate == 16_000, "Sample rate should be 16_000"
            audio = ta.transforms.Resample(orig_freq=sr,
                                           new_freq=sample_rate).to(device)(audio)

        encoded = unit_extractor(audio)
        unit, duration = process_unit(encoded, sample_rate, hop_size)

        np.save(os.path.join(base_units, os.path.basename(filepath).replace(".wav", ".npy")), unit)
        np.save(os.path.join(base_durations, os.path.basename(filepath).replace(".wav", ".npy")), duration)



@hydra.main(config_path="conf", config_name="config")
def main(cfg: TrainingUnitEncoderConfig_STEP1):
    os.chdir(get_original_cwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading speaker embedder ...")
    spkr_embedder = ECAPA_TDNN(feat_dim=cfg.spkr_embedder.feat_dim,
                            channels=cfg.spkr_embedder.channels,
                            emb_dim=cfg.spkr_embedder.spk_emb_dim,
                            feat_type=cfg.spkr_embedder.feat_type,
                            sr=cfg.spkr_embedder.sr,
                            feature_selection=cfg.spkr_embedder.feature_selection,
                            update_extract=cfg.spkr_embedder.update_extract,
                            config_path=cfg.spkr_embedder.config_path).to(device)
    if not os.path.exists(cfg.spkr_embedder.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint for speaker embedding extractor not found: {cfg.spkr_embedder.checkpoint}")
    state_dict = torch.load(cfg.spkr_embedder.checkpoint,
                            map_location=lambda loc, storage: loc)
    spkr_embedder.load_state_dict(state_dict["model"], strict=False)
    _ = spkr_embedder.eval()  # FREEZING SPEAKER EMBEDDER
    for param in spkr_embedder.parameters():
        param.requires_grad = False

    logging.info("Loading unit extractor...")
    unit_extractor = SpeechEncoder.by_name(dense_model_name=cfg.unit_extractor.dense_model_name,
                                           quantizer_model_name=cfg.unit_extractor.quantizer_name,
                                           vocab_size=cfg.unit_extractor.vocab_size,
                                           deduplicate=cfg.unit_extractor.deduplicate,
                                           need_f0=cfg.unit_extractor.need_f0).to(device)
    # Freeze the unit extractor
    _ = unit_extractor.eval()
    for param in unit_extractor.parameters():
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
    # generate_mels(filelist_path = cfg.dataset.train_filelist_path,
    #               **mels_config)
    # generate_mels(filelist_path = cfg.dataset.test_filelist_path,
    #               **mels_config)

    generate_spkr_embeddings(filelist_path=cfg.dataset.train_filelist_path,
                             spkr_embedder=spkr_embedder,
                             sample_rate=cfg.spkr_embedder.sr,
                             device=device)
    generate_spkr_embeddings(filelist_path=cfg.dataset.test_filelist_path,
                             spkr_embedder=spkr_embedder,
                             sample_rate=cfg.spkr_embedder.sr,
                             device=device)

    generate_unit_duration_embeddings(filelist_path=cfg.dataset.train_filelist_path,
                                      unit_extractor=unit_extractor,
                                      sample_rate=cfg.spkr_embedder.sr,
                                      hop_size=cfg.data.hop_length,
                                      device=device)
    generate_unit_duration_embeddings(filelist_path=cfg.dataset.test_filelist_path,
                                      unit_extractor=unit_extractor,
                                      sample_rate=cfg.spkr_embedder.sr,
                                      hop_size=cfg.data.hop_length,
                                      device=device)



if __name__ == "__main__":
    main()