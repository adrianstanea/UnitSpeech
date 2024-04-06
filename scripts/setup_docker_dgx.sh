#!/bin/bash

CONTAINER_NAME="UnitSpeech_1_0"
SOURCE_CODE_MOUNT="$(pwd)":/workspace/local
LJSPEECH_MOUNT="/mnt/QNAP/staria/LJSpeech-1.1/wavs:/datasets/LJSpeech"
OUTPUTS_MOUNT="/mnt/QNAP/staria/bogdan_outputs/UnitSpeech_Train1:/outputs"
PRETRAINER_CHECKPTS="/mnt/QNAP/staria/bogdan_outputs/pretrained-checkpoints:/checkpoints"

docker container run -d \
                    -it \
                    --name $CONTAINER_NAME \
                    -v $SOURCE_CODE_MOUNT \
                    -v $LJSPEECH_MOUNT \
                    -v $OUTPUTS_MOUNT \
                    -v $PRETRAINER_CHECKPTS \
                    --gpus=all \
                    nvcr.io/nvidia/pytorch:24.02-py3

docker exec -it $CONTAINER_NAME /bin/bash

# Delete a container
# docker container rm -f $CONTAINER_NAME