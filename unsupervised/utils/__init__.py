import torch
import numpy as np
import random
from .eval_utils import *
from .seqloss import SequenceLoss
def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True