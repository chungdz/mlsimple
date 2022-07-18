import json
import pickle
import numpy as np
import os

class NNConfig():
    def __init__(self, dpath='data'):

        self.hidden = 100
        self.emb_size = 10
        self.dpath = dpath
        self.meta = json.load(open(os.path.join(dpath, 'meta_info.json'), 'r'))
