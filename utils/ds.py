from torch.utils.data.dataset import Dataset, IterableDataset
import torch
import os
import pandas as pd
import numpy as np
import pandas as pd
import random

class ClassificationTrainDS(IterableDataset):
    def __init__(self, cfg, headerp, filep, chunk_size, isTrain=True):
        super(ClassificationTrainDS).__init__()
        
        self.header = pd.read_csv(headerp, sep='\t')
        self.filep = filep
        self.isTrain = isTrain
        
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
        self.chunk_size = chunk_size     
    
    def init_reader(self):
        self.dfiter = iter(pd.read_csv(self.filep, sep='\t', names=self.header.columns, iterator=True, chunksize=self.chunk_size))

    def __iter__(self):
        self.init_reader()
        worker_info = torch.utils.data.get_worker_info()
        while True:
            try:
                cur_chunk = next(self.dfiter)
            except:
                break
            
            for findex, fname in enumerate(self.flist):
                cur_chunk[fname] = (cur_chunk[fname] - self.to_sub[findex]) / self.to_div[findex]
                
            for dindex, idname in enumerate(self.idlist):
                if self.isTrain:
                    cur_chunk[idname] = cur_chunk[idname].apply(lambda x: self.dicts[dindex].get(str(x), 0) if random.random() > 0.006 else 0)
                else:
                    cur_chunk[idname] = cur_chunk[idname].apply(lambda x: self.dicts[dindex].get(str(x), 0))
                
            finputs = cur_chunk[self.flist].values
            idinputs = cur_chunk[self.idlist].values
            targets = cur_chunk["m:Click"].values
            
            if not worker_info is None:  # single-process data loading, return the full iterator
                assert(worker_info.num_workers == 1)
            
            yield {
                    "finputs": torch.FloatTensor(finputs),
                    "idinputs": torch.LongTensor(idinputs),
                    'labels': torch.FloatTensor(targets)
            }
        
        return

def collate_fn(batch):
    return {
        'finputs': torch.cat([x['finputs'] for x in batch], dim=0),
        "idinputs" : torch.cat([x['idinputs'] for x in batch], dim=0),
        'labels': torch.cat([x['labels'] for x in batch], dim=0)
    }
