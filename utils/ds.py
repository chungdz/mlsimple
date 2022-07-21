from torch.utils.data.dataset import Dataset
import torch
import os
import pandas as pd
import numpy as np
import pandas as pd

class ClassificationTrainDS(Dataset):
    def __init__(self, cfg, df):
        self.finputs = df[cfg.meta['features']].values
        self.idinputs = df[cfg.meta['ids']].values
        self.dicts = cfg.meta['dicts']

        self.to_sub = np.array(cfg.meta['to_minus'])
        self.to_div = np.array(cfg.meta['to_div'])
        self.target = df['m:Click'].values

    def __getitem__(self, index):
        finputs = (self.finputs[index] - self.to_sub) / self.to_div
        idinputs = np.array([self.dicts[i][str(cid)] for i, cid in enumerate(self.idinputs[index])])
        label = self.target[index]

        return {
            "finputs": torch.FloatTensor(finputs),
            "idinputs": torch.LongTensor(idinputs),
            'label': label
        }

    def __len__(self):
        return self.target.shape[0]

def collate_fn(batch):
    return {
        'finputs': torch.stack([x['finputs'] for x in batch]),
        "idinputs" : torch.stack([x['idinputs'] for x in batch]),
        'labels': torch.FloatTensor([x['label'] for x in batch])
    }