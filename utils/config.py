import json
import pickle
import numpy as np
import os

class NNConfig():
    def __init__(self, dpath='data'):

        self.hidden = 100
        self.dpath = dpath
