import numpy as np
import random
import torch


def setseed(seed):
    """Sets the seed for rng."""
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.random.manual_seed(seed)