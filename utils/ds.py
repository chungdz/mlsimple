from torch.utils.data.dataset import Dataset
import torch
import os
import pandas as pd
import numpy as np
import pandas as pd

class ClassificationTrainDS(Dataset):
    def __init__(self, cfg, df):
        self.finputs = df[cfg['features']].values
        self.idinputs = df[cfg['id_feature']].values
        self.dicts = cfg['idxdicts']
        self.minlist = np.array(cfg['minlist'])
        self.flen = np.arrat(cfg['maxlist']) - np.array(cfg['minlist'])
        self.target = df['m:Click'].values

    def __getitem__(self, index):
        finputs = (self.finputs[index] - self.minlist) / self.flen
        idinputs = np.array([self.dicts[i][cid] for i, cid in enumerate(self.idinputs[index])])
        label = self.target[index]

        return {
            "finputs": torch.FloatTensor(finputs),
            "idinputs": torch.LongTensor(idinputs),
            'label': label
        }

    def __len__(self):
        return self.ds.shape[0]

def collate_fn(batch):
    return {
        'finputs': torch.stack([x['finputs'] for x in batch]),
        "idinputs" : torch.stack([x['idinputs'] for x in batch]),
        'labels': torch.FloatTensor([x['label'] for x in batch])
    }