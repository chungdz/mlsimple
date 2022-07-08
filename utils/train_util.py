# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020/10/10


"""
import os
import random

import torch
import numpy as np


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
