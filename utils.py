import logging
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch

from hydra.utils import get_original_cwd


def create_symlink(source_dir_name: str, target_path: str):
    """
    Creates a symbolic link from the source path to the target path.

    Args:
        source_dir_name (str): The name of the directory in the local instance.
        target_path (str): The path in the host machine where a dataset is located.
    """
    source_path = Path(get_original_cwd()) / source_dir_name

    logging.debug(f"Creating symlink from {source_path} to {target_path}")

    # Check if the symlink already exists
    if not os.path.islink(source_path):
        if os.path.exists(source_path):
            logging.info(f"File or directory {source_path} already exists and is not a symlink.")
        else:
            os.symlink(target_path, str(source_path))
    else:
        logging.info(f"Symlink {source_path} already exists.")