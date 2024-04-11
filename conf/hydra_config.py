from dataclasses import dataclass, field
from optparse import Option
from typing import List, Any, Optional

from omegaconf import MISSING

from unitspeech.text.symbols import symbols
from unitspeech.util import fix_len_compatibility

# defaults = [
#     {"dataset": "LJSpeech"},
# ]


@dataclass
class DataConfig:
    n_units: int = 1000
    n_feats: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sampling_rate: int = 22050
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    cmudict_path: str = 'resources/cmu_dictionary'
    add_blank: bool = True

@dataclass
class LJSPeechConfig:
    train_filelist_path: str = 'resources/filelists/ljspeech/train.txt'
    test_filelist_path: str = 'resources/filelists/ljspeech/test.txt'
    # path: str = 'LJSpeech/wavs' # SERVER
    path: str = 'LJSpeech' # LOCAL

@dataclass
class LibriTTSConfig:
    train_filelist_path: str = 'resources/filelists/libri-tts/train.txt'
    test_filelist_path: str = 'resources/filelists/libri-tts/valid.txt'
    # path: str = 'LibriTTS/wavs' # Server
    path: str = 'LibriTTS' # LOCAL


@dataclass
class UnitEncoderConfig:
    n_channels: int = 192
    filter_channels: int = 768
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_heads: int = 2
    window_size: int = 4


@dataclass
class TextEncoderConfig:
    n_vocab: int = len(symbols)+1
    n_channels: int = 192
    filter_channels: int = 768
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_heads: int = 2
    window_size: int = 4


@dataclass
class DurationPredictorConfig:
    in_channels: int = 192
    filter_channels: int = 256
    kernel_size: int = 3
    p_dropout: float = 0.1
    spk_emb_dim: int = 256


@dataclass
class DecoderConfig:  # GradTTS -> diffusion model
    dim: int = 128
    dim_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pe_scale: int = 1000
    beta_min: float = 0.05
    beta_max: float = 20.0
    spk_emb_dim: int = 256
    diffusion_steps: int = 50
    checkpoint: str = "unitspeech/checkpoints/pretrained_decoder.pt"

    # def __post_init__(self):
    #     self.num_downsamplings_in_unet = len(self.dim_mults) - 1


@dataclass
class TrainConfig:
    cuda_device: bool = True
    out_size_second: int = 2
    n_epochs: int = 1000
    batch_size: int = 16
    drop_last: bool = True
    num_workers : int = 0
    shuffle : bool = True
    fp16_run: bool = False
    seed: int = 42
    log_dir: str = 'logs/new_exp'
    save_every: int = 1 # TODO: change to 50
    test_size: int = 4

    # def __post_init__(self):
    #     self.out_size = fix_len_compatibility(self.out_size_second * DataConfig.sampling_rate // DataConfig.hop_length,
    #                                           num_downsamplings_in_unet=DecoderConfig.num_downsamplings_in_unet)


@dataclass
class VocoderConfig:
    config_path: str = "unitspeech/vocoder/checkpts/bigvgan-config.json"
    ckpt_path: str = "unitspeech/vocoder/checkpts/bigvgan.pt"


@dataclass
class AdamConfig:
    learning_rate: float = 1e-4


@dataclass
class SpeakerEncoderCfg:
    feat_dim: int = 1024
    feat_type: str = "wavlm_large"
    config_path: Optional[str] = None
    channels:int  = 512
    spk_emb_dim: int = 256
    sr: int = 16000
    feature_selection: str = "hidden_states"
    update_extract: bool = False
    checkpoint: str = "unitspeech/checkpoints/speaker_encoder.pt" # Speaker Embedding Extractor

@dataclass
class UnitExtractorConfig:
    dense_model_name: str = "mhubert-base-vp_en_es_fr"
    quantizer_name: str = "kmeans"
    vocab_size : int = 1000 # Related to data.n_units
    deduplicate : bool = True
    need_f0 : bool = False

# Train Step 1 config => text encoder, duration predictor and decoder
@dataclass
class TrainingUnitEncoderConfig_STEP1:
    data: DataConfig = DataConfig()
    dataset : LJSPeechConfig = LJSPeechConfig()
    optimizer: AdamConfig = AdamConfig()
    vocoder: VocoderConfig = VocoderConfig()
    spkr_encoder: SpeakerEncoderCfg = SpeakerEncoderCfg()
    duration_predictor: DurationPredictorConfig = DurationPredictorConfig()
    unit_extractor: UnitExtractorConfig = UnitExtractorConfig()
    encoder: TextEncoderConfig = TextEncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    train: TrainConfig = TrainConfig()


# Train Step 2 config => Unit Encoder adaptation
@dataclass
class TrainingUnitEncoderConfig_STEP2:
    data: DataConfig = DataConfig()
    dataset : LJSPeechConfig = LJSPeechConfig()
    optimizer: AdamConfig = AdamConfig()
    vocoder: VocoderConfig = VocoderConfig()
    spkr_encoder: SpeakerEncoderCfg = SpeakerEncoderCfg()
    duration_predictor: DurationPredictorConfig = DurationPredictorConfig()
    unit_extractor: UnitExtractorConfig = UnitExtractorConfig()
    encoder: UnitEncoderConfig = UnitEncoderConfig() # Unit and Text encoders seem to have the same config
    decoder: DecoderConfig = DecoderConfig()
    train: TrainConfig = TrainConfig()
