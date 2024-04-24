#!/bin/bash

source ./variables.sh

# ln -s "$LJSPEECHT_CONTAINER_PATH" ../DUMMY
# ln -s "$LIBRITTS_CONTAINER_PATH/" ../DUMMY

# NOTE: delete checkpoints from source code before running this script
ln -s "$CHECKPOINTS_CONTAINER_PATH" ../unitspeech/checkpoints