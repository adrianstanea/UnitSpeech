import argparse
import logging
import os
import torch
import torchaudio as ta
from tqdm import tqdm
from conf.hydra_config import MainConfig
from unitspeech.textlesslib.textless.data.speech_encoder import SpeechEncoder
from unitspeech.util import parse_filelist, process_unit


def main(args):
    cfg = MainConfig
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.CUDA_VISIBLE_DEVICES
    logger.info(f"Using CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    logging.info("Loading unit extractor...")
    unit_extractor = SpeechEncoder.by_name(dense_model_name=cfg.unit_extractor.dense_model_name,
                                           quantizer_model_name=cfg.unit_extractor.quantizer_name,
                                           vocab_size=cfg.unit_extractor.vocab_size,
                                           deduplicate=cfg.unit_extractor.deduplicate,
                                           need_f0=cfg.unit_extractor.need_f0).to(device).eval()

    print(f"Loading filelist from {args.filelist_path}")
    filelist = parse_filelist(args.filelist_path, split_char='|')

    print(f"Number of samples: {len(filelist)}")
    for idx, line in tqdm(enumerate(filelist, start=1), total=len(filelist)):
        with torch.no_grad():
            filepath, text, spk_id = line[0], line[1], line[2]
            audio, sr = ta.load(filepath)
            audio = audio.to(device)
            if sr != cfg.spkr_embedder.sr:
                assert cfg.spkr_embedder.sr == 16_000, "Sample rate should be 16_000"
                audio = ta.transforms.Resample(orig_freq=sr,
                                            new_freq=cfg.spkr_embedder.sr).to(device)(audio)
            encoded = unit_extractor(audio)
            unit, duration = process_unit(encoded, cfg.spkr_embedder.sr, cfg.data.hop_length)

            parent_dir = os.path.dirname(filepath)
            filename = os.path.basename(filepath)
            basename, extension = os.path.splitext(filename)
            unit_base = f"{basename}_unit.pt"
            duration_base = f"{basename}_duration.pt"

            torch.save(unit, os.path.join(parent_dir, unit_base))
            torch.save(duration, os.path.join(parent_dir, duration_base))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist_path',
                    type=str,
                    default="resources/filelists/swara/metadata_SWARA1.0_text.csv")
    args = parser.parse_args()
    main(args)