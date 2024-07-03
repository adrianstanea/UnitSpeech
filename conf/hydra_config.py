from dataclasses import dataclass, field
from typing import List, Optional

from unitspeech.text.symbols import symbols


@dataclass
class InferenceConfig:
    ID: int = -10
    text: str = " Am citit 25 de pagini din carte"
    spkr_embs_path: str = "unitspeech/checkpoints/inference/spkr_embs/"
    with_plot: bool = True
    with_sv56_normalization: bool = True
    diffusion_steps: int = 50
    length_scale: float = 1.0
    text_gradient_scale: float = 1.0
    spk_gradient_scale: float = 1.0
    language: str = "ro"
    file_path: str = "audio.wav"
    use_finetuned_decoder: bool = True


@dataclass
class FinetuneConfig:
    reference_sample: str = "reference.wav"
    finetuned_decoders_path: str = "unitspeech/checkpoints/inference"
    ID: int = -1 
    learning_rate: float = 2e-5
    n_iters: int = 500


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
    cmudict_path: str = "resources/cmu_dictionary"
    add_blank: bool = True
    embs_path: str = "unitspeech/checkpoints/spkr_embs/"


@dataclass
class LJSPeechConfig:
    train_filelist_path: str = "resources/filelists/ljspeech/train.txt"
    test_filelist_path: str = "resources/filelists/ljspeech/test.txt"
    normalize_mels: bool = True
    mel_min_path: str = "unitspeech/checkpoints/mel_normalization/LJSpeech/mel_min.pt"
    mel_max_path: str = "unitspeech/checkpoints/mel_normalization/LJSpeech/mel_max.pt"
    text_uncon_path: str = "unitspeech/checkpoints/CFG/LJSpeech/text_uncond.pt"
    spk_uncond_path: str = "unitspeech/checkpoints/CFG/LJSpeech/spk_uncond.pt"
    name: str = "LJSpeech"


@dataclass
class LibriTTSConfig:
    train_filelist_path: str = "resources/filelists/libri-tts/train.txt"
    test_filelist_path: str = "resources/filelists/libri-tts/valid.txt"
    normalize_mels: bool = True
    mel_min_path: str = "unitspeech/checkpoints/mel_normalization/LibriTTS/mel_min.pt"
    mel_max_path: str = "unitspeech/checkpoints/mel_normalization/LibriTTS/mel_max.pt"
    text_uncond_path: str = "unitspeech/checkpoints/CFG/LibriTTS/text_uncond.pt"
    spk_uncond_path: str = "unitspeech/checkpoints/CFG/LibriTTS/spk_uncond.pt"
    name: str = "LibriTTS"


@dataclass
class SWARAConfig:
    train_filelist_path: str = "resources/filelists/swara/train.txt"
    test_filelist_path: str = "resources/filelists/swara/test.txt"
    normalize_mels: bool = True
    mel_min_path: str = "unitspeech/checkpoints/mel_normalization/SWARA/mel_min.pt"
    mel_max_path: str = "unitspeech/checkpoints/mel_normalization/SWARA/mel_max.pt"
    text_uncond_path: str = "unitspeech/checkpoints/CFG/SWARA/text_uncond.pt"
    spk_uncond_path: str = "unitspeech/checkpoints/CFG/SWARA/spk_uncond.pt"
    name: str = "SWARA"


@dataclass
class UnitEncoderConfig:
    n_channels: int = 192
    filter_channels: int = 768
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_heads: int = 2
    window_size: int = 4
    checkpoint: str = "unitspeech/checkpoints/unit_encoder.pt"
    train_checkpoint: str = "unitspeech/checkpoints/train/unit_encoder.pt"


@dataclass
class TextEncoderConfig:
    n_vocab: int = len(symbols) + 1
    n_channels: int = 192
    filter_channels: int = 768
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_heads: int = 2
    window_size: int = 4
    checkpoint: str = "unitspeech/checkpoints/text_encoder.pt"
    train_checkpoint: str = "unitspeech/checkpoints/train/text_encoder.pt"


@dataclass
class DurationPredictorConfig:
    in_channels: int = 192
    filter_channels: int = 256
    kernel_size: int = 3
    p_dropout: float = 0.1
    spk_emb_dim: int = 256
    checkpoint: str = "unitspeech/checkpoints/duration_predictor.pt"
    train_checkpoint: str = "unitspeech/checkpoints/train/duration_predictor.pt"


@dataclass
class DecoderConfig:  # GradTTS -> diffusion model
    dim: int = 128
    dim_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pe_scale: int = 1000
    beta_min: float = 0.05
    beta_max: float = 20.0
    spk_emb_dim: int = 256
    diffusion_steps: int = 500
    checkpoint: str = "unitspeech/checkpoints/pretrained_decoder.pt"
    train_checkpoint: str = "unitspeech/checkpoints/train/pretrained_decoder.pt"


@dataclass
class TrainConfig:
    CUDA_VISIBLE_DEVICES: str = "7"
    on_GPU: bool = True
    out_size_second: int = 2
    n_epochs: int = 2000
    batch_size: int = 32
    drop_last: bool = True
    num_workers: int = 4
    shuffle: bool = True
    fp16_run: bool = False
    seed: int = 42
    log_dir: str = "logs/new_exp"
    save_every: int = 5
    test_size: int = 4
    from_checkpoint: bool = True
    with_uncond_score_estimator: bool = True


@dataclass
class VocoderConfig:
    config_path: str = "unitspeech/vocoder/checkpts/bigvgan-config.json"
    ckpt_path: str = "unitspeech/vocoder/checkpts/bigvgan.pt"


@dataclass
class AdamConfig:
    learning_rate: float = 1e-4  # TRAIN
    # learning_rate: float = 1e-5 # FINETUNE


@dataclass
class SpeakerEmbedderCfg:
    feat_dim: int = 1024
    feat_type: str = "wavlm_large"
    config_path: Optional[str] = None
    channels: int = 512
    spk_emb_dim: int = 256
    sr: int = 16000
    feature_selection: str = "hidden_states"
    update_extract: bool = False
    checkpoint: str = "unitspeech/checkpoints/speaker_encoder.pt"  # Speaker Embedding Extractor


@dataclass
class UnitExtractorConfig:
    dense_model_name: str = "mhubert-base-vp_en_es_fr"
    quantizer_name: str = "kmeans"
    vocab_size: int = 1000  # Related to data.n_units
    deduplicate: bool = True
    need_f0: bool = False


@dataclass
class MainConfig:
    data: DataConfig = DataConfig()
    dataset: SWARAConfig = SWARAConfig()
    optimizer: AdamConfig = AdamConfig()
    vocoder: VocoderConfig = VocoderConfig()
    spkr_embedder: SpeakerEmbedderCfg = SpeakerEmbedderCfg()
    duration_predictor: DurationPredictorConfig = DurationPredictorConfig()
    unit_extractor: UnitExtractorConfig = UnitExtractorConfig()
    text_encoder: TextEncoderConfig = TextEncoderConfig()
    unit_encoder: UnitEncoderConfig = UnitEncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    train: TrainConfig = TrainConfig()
    inference: InferenceConfig = InferenceConfig()
    finetune: FinetuneConfig = FinetuneConfig()
