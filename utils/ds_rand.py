from multidict import istr
from torch.utils.data.dataset import Dataset
import torch
import os
import pandas as pd
import numpy as np
import random
import pandas as pd

class ClassificationTrainDS(Dataset):
    def __init__(self, cfg, headerp, filep, isTrain=True):
        self.header = pd.read_csv(headerp, sep='\t')
        cur_chunk = pd.read_csv(filep, sep='\t', names=self.header.columns)
        
        dicts = []
        to_sub = []
        to_div = []
        self.flist = cfg.flist
        self.idlist = cfg.idlist
        
        for fname in self.flist:
            findex = cfg.meta['all_features'].index(fname)
            to_sub.append(cfg.meta['to_minus'][findex])
            to_div.append(cfg.meta['to_div'][findex])
        
        for idname in self.idlist:
            dindex = cfg.meta['all_ids'].index(idname)
            dicts.append(cfg.meta['dicts'][dindex])
        
        self.to_sub = to_sub
        self.to_div = to_div
        self.dicts = dicts
        
        for findex, fname in enumerate(self.flist):
            cur_chunk[fname] = (cur_chunk[fname] - self.to_sub[findex]) / self.to_div[findex]
                
        for dindex, idname in enumerate(self.idlist):
            if isTrain:
                cur_chunk[idname] = cur_chunk[idname].apply(lambda x: self.dicts[dindex].get(str(x), 0) if random.random() > 0.006 else 0)
            else:
                cur_chunk[idname] = cur_chunk[idname].apply(lambda x: self.dicts[dindex].get(str(x), 0))

        self.finputs = cur_chunk[self.flist].values      
        self.idinputs = cur_chunk[self.idlist].values
        self.targets = cur_chunk["m:Click"].values
        self.indexes = cur_chunk.index.values

    def __getitem__(self, index):
        return {
            "finputs": torch.FloatTensor(self.finputs[index]),
            "idinputs": torch.LongTensor(self.idinputs[index]),
            'labels': self.targets[index],
            'indexes': self.indexes[index]
        }

    def __len__(self):
        return self.targets.shape[0]

def collate_fn(batch):
    return {
        'finputs': torch.stack([x['finputs'] for x in batch]),
        "idinputs" : torch.stack([x['idinputs'] for x in batch]),
        'labels': torch.FloatTensor([x['labels'] for x in batch]),
        'indexes': torch.FloatTensor([x['indexes'] for x in batch])
    }