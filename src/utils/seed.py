"""Random seed helpers for reproducible experiments."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set supported random seeds without hard-failing on optional deps."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ModuleNotFoundError:
        pass
