from torch.utils.data.dataset import Dataset
import torch
import os
import pandas as pd
import numpy as np
import pandas as pd

class ClassificationTrainDS(Dataset):
    def __init__(self, cfg, headerp, filep):
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
            cur_chunk[idname] = cur_chunk[idname].apply(lambda x: self.dicts[dindex][str(x)])

        self.finputs = cur_chunk[self.flist].values      
        self.idinputs = cur_chunk[self.idlist].values
        self.targets = cur_chunk["m:Click"].values
                

    def __getitem__(self, index):
        return {
            "finputs": torch.FloatTensor(self.finputs[index]),
            "idinputs": torch.LongTensor(self.idinputs[index]),
            'label': self.targets[index]
        }

    def __len__(self):
        return self.target.shape[0]

def collate_fn(batch):
    return {
        'finputs': torch.stack([x['finputs'] for x in batch]),
        "idinputs" : torch.stack([x['idinputs'] for x in batch]),
        'labels': torch.FloatTensor([x['label'] for x in batch])
    }