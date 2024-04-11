# Train the Network

## Environment Setup

* Create a docker image using ./scripts/setup_docker_dgx.sh
  * The docker image need a volume mount to access the train dataset, pretrained models and the output directory
* Within the docker image setup Miniconda and create a virtual environment with Python 3.8
* Install the required packages using the command `pip install -r requirements.txt`
* Install the setup.py file using the command `pip install -e .`

## Training

* Train can be done with features computed at runtime or with precomputed features using the `process_dataset.py`
