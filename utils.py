import logging
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch

from hydra.utils import get_original_cwd


def create_symlink(local_dir_name: str, host_dataset_path: str):
    """
    Creates a symlink from a directory in the local instance to a directory in the host machine.

    Args:
        local_dir_name (str): The name of the directory in the local instance.
        host_dataset_path (str): The path in the host machine where a directory is located.
    """
    local_path = Path(get_original_cwd()) / local_dir_name

    logging.debug(f"Creating symlink from {local_path} to {host_dataset_path}")

    # Check if the symlink already exists
    if os.path.islink(local_path):
        os.unlink(local_path)
        logging.info(f"Removed existing symlink {local_path}")

    os.symlink(host_dataset_path, str(local_path))
    logging.info(f"Created symlink {local_path} -> {host_dataset_path}")