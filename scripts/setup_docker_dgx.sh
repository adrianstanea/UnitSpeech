#!/bin/bash

GPU_ID=6

CONTAINER_NAME="UnitSpeech_1_0"
SOURCE_CODE_MOUNT="$(pwd)":/workspace/local
# Mount: host_path:container_path
LJSPEECH_MOUNT="/mnt/QNAP/staria/LJSpeech-1.1/wavs:/datasets/LJSpeech"
OUTPUTS_MOUNT="/mnt/QNAP/staria/bogdan_outputs/UnitSpeech_Train1:/outputs"
PRETRAINED_CHECKPTS="/mnt/QNAP/staria/bogdan_outputs/pretrained-checkpoints:/checkpoints"

docker container run -d \
                    -it \
                    --name $CONTAINER_NAME \
                    -v $SOURCE_CODE_MOUNT \
                    -v $LJSPEECH_MOUNT \
                    -v $OUTPUTS_MOUNT \
                    -v $PRETRAINED_CHECKPTS \
                    -p 42509:42509 \
                    --gpus=all \
                    nvcr.io/nvidia/pytorch:24.02-py3

docker exec -it -e CUDA_VISIBLE_DEVICES=$GPU_ID $CONTAINER_NAME /bin/bash

# Delete a container
# docker container rm -f $CONTAINER_NAME

# Experimental: Forward port from container to host and then connect tensorboard from local machine to the container