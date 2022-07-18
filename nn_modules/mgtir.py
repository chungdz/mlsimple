import torch
import torch.nn as nn
import torch.nn.functional as F

class MGTIR(nn.Module):

    def __init__(self, cfg):
        super(MGTIR, self).__init__()
        
        self.hidden = cfg.hidden
        self.seq = nn.Sequential(
            nn.Linear(12, self.hidden),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.idlen * cfg.emb_size, self.hidden // 2),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Linear(self.hidden + self.hidden // 2, 1),
            nn.Sigmoid())
        self.embLayer = nn.ModuleList([nn.Embedding(len(d), cfg.emb_size) for d in cfg.meta['dicts']])
        self.idlen = len(cfg.meta['dicts'])
        
        
    def predict(self, finputs, idinputs):
        embs = []
        for i in range(self.idlen):
            embs.append(self.embLayer(idinputs[:, i]))
        embt = torch.cat(embs, dim=-1)
        concated = torch.cat([self.seq1(finputs), self.seq2(embt)], dim=-1)
        return self.seq3(concated)

    def forward(self, finputs, idinputs, labels):

        logits = self.seq(finputs, idinputs)
        loss = F.binary_cross_entropy(logits.squeeze(), labels)

        return {
            'loss': loss,
            'logits': logits
        }


    

