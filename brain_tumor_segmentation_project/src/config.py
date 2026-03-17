import random

import numpy as np
import torch

DEFAULT_DATASET_NAME = "dwb2023/brain-tumor-image-dataset-semantic-segmentation"
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
