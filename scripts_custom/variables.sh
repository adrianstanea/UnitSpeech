#!/bin/bash

LJSPEECH_HOST_PATH="/mnt/QNAP/staria/LJSpeech-1.1/wavs"
LJSPEECHT_CONTAINER_PATH="/datasets/LJSpeech"

LIBRITTS_HOST_PATH="/mnt/QNAP/staria/LibriTTS"
LIBRITTS_CONTAINER_PATH="/datasets/LibriTTS"

SWARA_HOST_PATH="/mnt/QNAP/staria/SWARA/SWARA1.0_22k_noSil"
# Link path in container according to metadata file
SWARA_CONTAINER_PATH="/media/DATA/CORPORA/SWARA2.0/SWARA1.0_22k/"

OUTPUTS_HOST_PATH="/mnt/QNAP/staria/Bogdan/train_logs/UnitSpeech_Train3"
OUTPUTS_CONTAINER_PATH="/outputs"

CHECKPOINTS_HOST_PATH="/mnt/QNAP/staria/Bogdan/unitspeech/checkpoints"
CHECKPOINTS_CONTAINER_PATH="/checkpoints"