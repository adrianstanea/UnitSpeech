#!/bin/bash

source ./variables.sh

cd ..

GPU_ID=6
IMAGE="nvcr.io/nvidia/pytorch:23.06-py3"

CONTAINER_NAME="UnitSpeech-licenta-RO"
SOURCE_CODE_MOUNT="$(pwd)":/workspace/local

LJSPEECH_MOUNT="$LJSPEECH_HOST_PATH:$LJSPEECHT_CONTAINER_PATH"
LIBRITTS_MOUNT="$LIBRITTS_HOST_PATH:$LIBRITTS_CONTAINER_PATH"
SWARA_MOUNT="$SWARA_HOST_PATH:$SWARA_CONTAINER_PATH"
OUTPUTS_MOUNT="$OUTPUTS_HOST_PATH:$OUTPUTS_CONTAINER_PATH"
CHECKPOINTS_MOUNT="$CHECKPOINTS_HOST_PATH:$CHECKPOINTS_CONTAINER_PATH"

UID=$(id -u)
GID=$(id -g)

docker container run -d \
                    -it \
                    --name $CONTAINER_NAME \
                    -v $SOURCE_CODE_MOUNT \
                    -v $LJSPEECH_MOUNT \
                    -v $LIBRITTS_MOUNT \
                    -v $SWARA_MOUNT \
                    -v $OUTPUTS_MOUNT \
                    -v $CHECKPOINTS_MOUNT \
                    --gpus=all \
                    $IMAGE

docker exec -it -e CUDA_VISIBLE_DEVICES=$GPU_ID $CONTAINER_NAME /bin/bash

# Delete a container
# docker container rm -f $CONTAINER_NAME

# Experimental: Forward port from container to host and then connect tensorboard from local machine to the container