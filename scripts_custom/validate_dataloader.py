
from torch.utils.data import DataLoader

import os
import hydra
from conf.hydra_config import AdamConfig, TrainingUnitEncoderConfig
from hydra.utils import get_original_cwd, to_absolute_path
from hydra.core.config_store import ConfigStore

import logging

from data.data import UnitMelSpeakerBatchCollate, UnitMelSpeakerDataset
from unitspeech.encoder import Encoder
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder


logger = logging.getLogger("train_unit-encoder.py")
logger.setLevel(logging.INFO)

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig)


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig):
    os.chdir(get_original_cwd())
    logger.debug(f"Running from: {os.getcwd()}")
    logger.info(f"logging data into: {cfg.train.log_dir}")

    logger.info("Initializing UnitExtractor...")
    unit_extractor = SpeechEncoder.by_name(dense_model_name=cfg.unit_extractor.dense_model_name,
                                           quantizer_model_name=cfg.unit_extractor.quantizer_name,
                                           vocab_size=cfg.unit_extractor.vocab_size,
                                           deduplicate=cfg.unit_extractor.deduplicate,
                                           need_f0=cfg.unit_extractor.need_f0)
    _ = unit_extractor.cuda().eval()
    
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

    train_batch = train_dataset.sample_test_batch(size=4)
    print(f"train_batch: {train_batch.shape}")
    for item in train_batch:
        mel, unit, spkr_id = item["y"], item["x"], item["spkr"]


if __name__ == "__main__":
    hydra_main()
