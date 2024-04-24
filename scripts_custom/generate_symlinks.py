# DEPRECATED !!
import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

from conf.hydra_config import TrainingUnitEncoderConfig_STEP1
from utils import create_symlink

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingUnitEncoderConfig_STEP1)

@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: TrainingUnitEncoderConfig_STEP1):
    os.chdir(get_original_cwd())

    logging.info("Creating a symlink to dataset...")
    source_dir_name = "DUMMY"
    print(cfg.dataset.path)
    target_path = os.path.join('/datasets', cfg.dataset.path)
    create_symlink(source_dir_name, target_path)

    source_dir_name = "unitspeech/checkpoints"
    target_path = "/checkpoints"
    create_symlink(source_dir_name, target_path)


if __name__ == "__main__":
    hydra_main()
