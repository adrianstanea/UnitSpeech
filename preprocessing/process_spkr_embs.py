import argparse
import gc
import logging
import os
import torch
from tqdm import tqdm
from conf.hydra_config import MainConfig
from preprocessing.utils import load_and_process_wav
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN
from unitspeech.util import parse_filelist

count = 0


def main(args):
    global count
    cfg = MainConfig
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.train.CUDA_VISIBLE_DEVICES
    logger.info(f"Using CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    spkr_embedder = (
        ECAPA_TDNN(
            feat_dim=cfg.spkr_embedder.feat_dim,
            channels=cfg.spkr_embedder.channels,
            emb_dim=cfg.spkr_embedder.spk_emb_dim,
            feat_type=cfg.spkr_embedder.feat_type,
            sr=cfg.spkr_embedder.sr,
            feature_selection=cfg.spkr_embedder.feature_selection,
            update_extract=cfg.spkr_embedder.update_extract,
            config_path=cfg.spkr_embedder.config_path,
        )
        .to(device)
        .eval()
    )

    state_dict = torch.load(cfg.spkr_embedder.checkpoint, map_location=lambda loc, storage: loc)
    spkr_embedder.load_state_dict(state_dict["model"], strict=False)
    for param in spkr_embedder.parameters():
        param.requires_grad = False

    filelist = parse_filelist(args.filelist_path, split_char="|")
    print(f"Loading filelist from {args.filelist_path}")

    embs_save_path = os.path.join(cfg.data.embs_path, cfg.dataset.name)
    if not os.path.exists(embs_save_path):
        print(f"Created directory {embs_save_path}")
        os.makedirs(embs_save_path)
    else:
        print(f"Directory {embs_save_path} already exists")

    crnt_speaker = -1
    num_samples = 0

    crnt_spkr_mean = torch.zeros(cfg.spkr_embedder.spk_emb_dim).unsqueeze(0).to(device)
    is_first_sample = True

    print(f"Number of samples: {len(filelist)}")
    for idx, line in tqdm(enumerate(filelist, start=1), total=len(filelist)):
        with torch.no_grad():
            filepath, text, spk_id = line[0], line[1], line[2]

            if crnt_speaker == -1:
                print(f"First speaker was: {spk_id}")
                crnt_speaker = spk_id

            # New speaker detected or last sample or to many samples
            if spk_id != crnt_speaker or idx == len(filelist):
                # Save data from old speaker - expect samples to be contiguous
                print(f"Number of samples: {num_samples} for speaker {crnt_speaker}")
                print(f"{crnt_speaker} | {crnt_spkr_mean}")
                # Save previous speaker's mean embedding
                save_path = os.path.join(embs_save_path, f"{crnt_speaker}.pt")
                torch.save(crnt_spkr_mean, save_path)
                # Reset for new speaker ID
                del crnt_spkr_mean
                gc.collect()
                crnt_spkr_mean = torch.zeros(cfg.spkr_embedder.spk_emb_dim).unsqueeze(0).to(device)
                is_first_sample = True
                num_samples = 0
                count = 0
                crnt_speaker = spk_id
                print(f"Speaker change to {spk_id} | current line = {idx}")
            wav = load_and_process_wav(filepath, device)
            cnrt_embedding = spkr_embedder(wav.unsqueeze(0))
            if is_first_sample:
                crnt_spkr_mean = cnrt_embedding
                is_first_sample = False
                count = 0
            else:
                crnt_spkr_mean = (crnt_spkr_mean * count + cnrt_embedding) / (count + 1)
            num_samples += 1
            count += 1
            del wav
            del cnrt_embedding
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist_path", type=str, default="resources/filelists/swara/metadata_SWARA_ALL_2.csv")
    parser.add_argument("--embs_save_path", type=str, default="unitspeech/checkpoints/spkr_embs/SWARA/")

    args = parser.parse_args()
    main(args)
