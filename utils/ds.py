from torch.utils.data.dataset import Dataset
import torch
import os
import pandas as pd
import numpy as np
import pandas as pd

class ClassificationTrainDS(Dataset):
    def __init__(self, cfg):
        df = pd.read_csv(os.path.join(cfg.dpath, 'test_small.tsv'), sep='\t')
        self.ds = df.values[:, 10:].astype(float)
        self.target = df.values[:, 0].astype(int)

    def __getitem__(self, index):
        inputs = self.ds[index]
        label = self.target[index]

        return {
            "inputs": torch.FloatTensor(inputs),
            'label': label
        }

    def __len__(self):
        return self.ds.shape[0]

def collate_fn(batch):
    return {
        'inputs': torch.stack([x['inputs'] for x in batch]),
        'labels': torch.FloatTensor([x['label'] for x in batch])
    }