#!/bin/bash

source ./variables.sh
cd ..

GPU_ID=6
IMAGE="nvcr.io/nvidia/pytorch:23.06-py3"
CONTAINER_NAME="UnitSpeech-licenta"
SOURCE_CODE_MOUNT="$(pwd)":/workspace/local
SWARA_MOUNT2="$SWARA_HOST_PATH2:$SWARA_CONTAINER_PATH2"
OUTPUTS_MOUNT="$OUTPUTS_HOST_PATH:$OUTPUTS_CONTAINER_PATH"
CHECKPOINTS_MOUNT="$CHECKPOINTS_HOST_PATH:$CHECKPOINTS_CONTAINER_PATH"

docker container run -d \
    -it \
    --name $CONTAINER_NAME \
    -v $SOURCE_CODE_MOUNT \
    -v $SWARA_MOUNT2 \
    -v $OUTPUTS_MOUNT \
    -v $CHECKPOINTS_MOUNT \
    --gpus=all \
    $IMAGE
docker exec -it -e CUDA_VISIBLE_DEVICES=$GPU_ID $CONTAINER_NAME /bin/bash
