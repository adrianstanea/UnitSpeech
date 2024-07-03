## UnitSpeech: Speaker-adaptive Speech Synthesis with Untranscribed Data
### [Reference Paper](https://arxiv.org/abs/2306.16083)

## Updates

- Normalize the audio of the synthesized speech files using the [sv56 executable from the G722 ITU recommandation](https://github.com/foss-for-synopsys-dwc-arc-processors/G722/tree/master/sv56)

- Create train scripts
- Adapt fine-tuning and inference scripts to the new structure using Hydra

## Installation

**Tested on Ubuntu 22.04.4 LTS, Python 3.8, Miniconda environment**

- Clone the git repository for this project:

```shell
git clone https://github.com/adrianstanea/UnitSpeech
```

- Enter the `scripts_custom` directory and run the following bash script to build the Docker container. Make sure to adapt the `variables.sh` file to your needs.

```shell
./setup_docker_dgx.sh
```

- Install the Miniconda package manager following the instructions on the [official website](https://docs.conda.io/en/latest/miniconda.html) for the Linux OS.<br>

- Install the necessary package for the IPA phonemizer.

```shell
sudo apt-get install espeak espeak-ng
```

- Build the sv56 audio normalization executable from the G722 ITU recommandation and place it under a reachable path.

```shell
git clone https://github.com/foss-for-synopsys-dwc-arc-processors/G722.git
cd G722/sv56/
make -f makefile.unx
cp sv56demo /bin
```

- After that, create a conda environment and install the unitspeech package and the package required for extracting speaker embeddings.

```shell
cd UnitSpeech
pip install -e .
pip install --no-deps s3prl==0.4.10
pip install -r requirements.txt
```

## Pretrained Models

**We provide the [pretrained models](https://drive.google.com/drive/folders/1yFkb2TAYB_zMmoTuUOXu-zXb3UI9pVJ9?usp=sharing).**
|File Name|Usage|
|------|---|
|unit_encoder.pt|Used for fine-tuning and unit-based speech synthesis tasks.<br>(e.g., Adaptive Speech Synthesis for Speech-to-Unit Translation)|
|text_encoder.pt|Used for adaptive text-to-speech tasks.|
|duration_predictor.pt|Used for adaptive text-to-speech tasks.|
|pretrained_decoder.pt|Used for all adaptive speech synthesis tasks.|
|speaker_encoder.pt|Used for extracting [speaker embeddings](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification#pre-trained-models).|
|bigvgan.pt|[Vocoder](https://github.com/NVIDIA/BigVGAN) checkpoint.|
|bigvgan-config.json|Configuration for the vocoder.|

**After downloading the files, please arrange them in the following structure.**

```buildoutcfg
UnitSpeech/...
    unitspeech/...
        checkpoints/...
            duration_predictor.pt
            pretrained_decoder.pt
            text_encoder.pt
            unit_encoder.pt
            CFG/
                [DATASET_NAME]/
                    ...                   -> Place unconditional mel and speaker features used in classifier-free guidance
            mel_normalization/
                [DATASET_NAME]/
                    ...                   -> Place normalization parameters extracted for the dataset
            spkr_embs/
                [DATASET_NAME]/
                    ...                   -> Place speaker embeddings extracted from the dataset
            train/
                ...                       -> Place checkpoint saved to resume training
            inference/
                <FINETUNED_DECODER_ID>.pt -> Placed by the finetune script and accessed by the inference script using the <ID> portion
            ...

            speaker_encoder.pt
            ...
        vocoder/...
            checkpts/...
                bigvgan.pt
                bigvgan-config.json
            ...
        ...
    ...
```

## Training

### Data Preparation

- Extract the necessary features from the dataset by executing the followings scripts within the `preprocessing` directory

```shell
python3 preprocessing/process_mel_normalization.py
python3 preprocessing/process_spkr_embs.py
python3 preprocessing/process_uncond_mel.py
python3 preprocessing/process_uncond_spk.py
python3 preprocessing/process_units.py
```

### Base models: text encoder, duration predictor and decoder

- Run the following script to train the base models: text encoder, duration predictor and decoder. Adjust the hyperparameters as needed. Training can be resumed from a checkpoint if the configuration is enabled within Hydra and the checkpoint is placed in the `checkpoints/train` directory.

```shell
python3 train_STEP1.py
```

### Unit encoder

```shell
python3 train_STEP2.py
```

## Fine-tuning

The decoder is fine-tuned using the target speaker's voice, employing the unit encoder. **It is recommended to use a reference English speech with a duration of at least 5~10 seconds.**

- The finetuned decoder will be saved in  

```shell
python3 finetune.py --ID=<ID> \
                    --n_iters=<N_ITERS> \
                    --learning_rate=<LEARNING_RATE> \
                    --reference_sample=reference-speech-BOGDAN.wav
```

By default, fine-tuning is conducted using the Adam optimizer with a learning rate of 2e-5 for 500 iterations.<br>
**For speakers with unique voices, increasing the number of fine-tuning iterations can help achieve better results.** <br>

## Inference

- The ID must match the one provided in the fine-tuning script.

```shell
python3 inference.py --ID=<ID> \
                     --generated_sample_path=<GENERATED_SAMPLE_PATH> \
                     --text=<TEXT> \
                     --diffusion_steps=<DIFFUSION_STEPS> \
                     --text_gradient_scale=<TEXT_GRADIENT_SCALE> \
                     --spk_gradient_scale=<SPK_GRADIENT_SCALE> \
                     --length_scale=<LENGTH_SCALE>
```

You can adjust the number of diffusion steps, length_scale, text gradient scale, and speaker gradient scale as arguments.<br>

- **text_gradient_scale** : responsible for pronunciation accuracy and audio quality. Increasing its value makes the pronunciation of the samples more accurate.<br>
- **spk_gradient_scale** : responsible for speaker similarity. Increasing its value generates voices that are closer to the reference speech.<br>

By default, text gradient scale is set to 1.0, and speaker gradient scale is set to 1.0.<br>
**If you want better pronunciation and audio quality, please increase the value of "text_gradient_scale." This will slightly reduce speaker similarity.**<br>
**If you want better speaker similarity, please increase the value of "spk_gradient_scale." This will slightly degrade pronunciation accuracy and audio quality.**<br>

You can adjust the speed of speaking as arguments. (default: 1.0) <br>

- **length_scale** : Increasing its value (> 1.0) makes the speech slow, while decreasing its value (< 1.0) makes the speech fast <br>

**Note: Using excessively large gradient scales can degrade the audio quality.**

## References

- This repository uses the source code from UnitSpeech and extends it to allow for training the base models using the Hydra framework. The original source code can be found at the following link:
  - [UnitSpeech](https://github.com/gmltmd789/UnitSpeech) (original source code implementation)
