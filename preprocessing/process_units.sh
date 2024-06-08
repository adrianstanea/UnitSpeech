#!/bin/bash

echo "Current working directory: $(pwd)"

cd ..

# LJSpeech
# file_paths=(
#     "resources/filelists/ljspeech/train.txt"
#     "resources/filelists/ljspeech/test.txt"
#     "resources/filelists/ljspeech/valid.txt"
# )

# LibriTTS
# file_paths=(
#     "resources/filelists/libri-tts/train.txt"
#     "resources/filelists/libri-tts/valid.txt"
# )

# SWARA
file_paths=(
    "resources/filelists/swara/metadata_SWARA1.0_text.csv"
)


echo "Processing units for the following file paths:"
for file_path in "${file_paths[@]}"; do
    echo "$file_path"
    python3 preprocessing/process_units.py --filelist_path "$file_path"
done
