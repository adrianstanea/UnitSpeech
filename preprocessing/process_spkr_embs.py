import argparse
import gc
import logging
import os

import librosa
import torch
import torchaudio

from conf.hydra_config import (
    TrainingUnitEncoderConfig_STEP1,
)
from preprocessing.utils import load_and_process_wav
from unitspeech.speaker_encoder.ecapa_tdnn import ECAPA_TDNN
from unitspeech.util import parse_filelist


def main(args):
    cfg = TrainingUnitEncoderConfig_STEP1
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.on_GPU else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    spkr_embedder = ECAPA_TDNN(feat_dim=cfg.spkr_embedder.feat_dim,
                        channels=cfg.spkr_embedder.channels,
                        emb_dim=cfg.spkr_embedder.spk_emb_dim,
                        feat_type=cfg.spkr_embedder.feat_type,
                        sr=cfg.spkr_embedder.sr,
                        feature_selection=cfg.spkr_embedder.feature_selection,
                        update_extract=cfg.spkr_embedder.update_extract,
                        config_path=cfg.spkr_embedder.config_path).to(device).eval()

    state_dict = torch.load(cfg.spkr_embedder.checkpoint,
                            map_location=lambda loc, storage: loc)
    spkr_embedder.load_state_dict(state_dict["model"], strict=False)
    for param in spkr_embedder.parameters():
        param.requires_grad = False

    filelist = parse_filelist(args.filelist_path, split_char='|')
    print(f"Loading filelist from {args.filelist_path}")

    if not os.path.exists(args.embs_save_path):
        print(f"Created directory {args.embs_save_path}")
        os.makedirs(args.embs_save_path)
    else:
        print(f"Directory {args.embs_save_path} already exists")
    
    crnt_speaker = -1
    num_samples = 0

    crnt_spkr_mean = torch.zeros(cfg.spkr_embedder.spk_emb_dim).unsqueeze(0).to(device)
    is_first_sample = True

    print(f"Number of samples: {len(filelist)}")
    for idx, line in enumerate(filelist, start=1):
        with torch.no_grad():
            filepath, text, spk_id = line[0], line[1], line[2]
        
            if idx % 100 == 0 or idx == 0:
                print(f"Processing line ({idx}|{len(filelist)})")
            
            # START FROM SPK_ID => used to skip to a specific speaker (currently this script crashes due to RAM overflow) 
            # if crnt_speaker == -1 and spk_id !="204":
            #     continue

            if crnt_speaker == -1:
                print(f"First speaker was: {spk_id}")
                crnt_speaker = spk_id

            # New speaker detected or last sample or to many samples
            if spk_id != crnt_speaker or idx==len(filelist): 
                # Save data from old speaker - expect samples to be contiguous
                print(f"Number of samples: {num_samples} for speaker {crnt_speaker}")
                # Save previous speaker's mean embedding
                save_path = os.path.join(args.embs_save_path, f"{crnt_speaker}.pt")
                torch.save(crnt_spkr_mean, save_path)

                # Reset for new speaker ID
                del crnt_spkr_mean
                gc.collect()
                crnt_spkr_mean = torch.zeros(cfg.spkr_embedder.spk_emb_dim).unsqueeze(0).to(device)
                is_first_sample = True
                num_samples = 0
                crnt_speaker = spk_id
                print(f"Speaker change to {spk_id} | current line = {idx}")

            wav = load_and_process_wav(filepath, device)
            cnrt_embedding = spkr_embedder(wav.unsqueeze(0))
            if is_first_sample:
                crnt_spkr_mean = cnrt_embedding
                is_first_sample = False
            else:
                crnt_spkr_mean = torch.mean(torch.stack([crnt_spkr_mean, cnrt_embedding]), dim=0)
                del wav
                del cnrt_embedding
            num_samples += 1
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # embs_save_path is used in the train script to load the speaker embeddings
    parser = argparse.ArgumentParser()

    # LibriTTS
    # parser.add_argument('--filelist_path',
    #                     type=str,
    #                     default='resources/filelists/libri-tts/train.txt')
    # parser.add_argument('--embs_save_path',
    #                 type=str,
    #                 default='unitspeech/checkpoints/spkr_embs/LibriTTS/')

    # LJSpeech
    # parser.add_argument('--filelist_path',
    #                     type=str,
    #                     default='resources/filelists/ljspeech/train.txt')
    # parser.add_argument('--embs_save_path',
    #                 type=str,
    #                 default='unitspeech/checkpoints/spkr_embs/LJSpeech/')

    args = parser.parse_args()
    main(args)