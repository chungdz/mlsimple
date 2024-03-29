import torch
import torch.nn as nn
import torch.nn.functional as F

class MGTIR(nn.Module):

    def __init__(self, cfg):
        super(MGTIR, self).__init__()
        
        self.hidden = cfg.hidden
        self.idlen = len(cfg.idlist)
        self.flen = len(cfg.flist)
        self.wd = cfg.weight_decay
        self.seq = nn.Sequential(
            nn.Linear(self.flen, self.hidden),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.idlen * cfg.emb_size, self.hidden // 2),
            nn.ReLU()
        )
        self.seq3 = nn.Linear(self.hidden + self.hidden // 2, 1)
        
        selected = []
        for idname in cfg.idlist:
            dindex = cfg.meta['all_ids'].index(idname)
            selected.append(cfg.meta['dicts'][dindex])
        self.embLayer = nn.ModuleList([nn.Embedding(len(d), cfg.emb_size) for d in selected])
        
    def predict(self, finputs, idinputs):
        embs = []
        for i in range(self.idlen):
            embs.append(self.embLayer[i](idinputs[:, i]))
        embt = torch.cat(embs, dim=-1)
        concated = torch.cat([self.seq(finputs), self.seq2(embt)], dim=-1)
        return self.seq3(concated)

    def forward(self, finputs, idinputs, labels, indexes):

        raw_logits = self.predict(finputs, idinputs)
        logits = torch.sigmoid(raw_logits)

        loss_weights = torch.clone(labels)
        loss_weights.masked_fill_(~loss_weights.bool(), self.wd)
        loss = F.binary_cross_entropy(logits.squeeze(), labels, weight=loss_weights)

        return {
            'loss': loss,
            'logits': logits,
            'raw': raw_logits,
            'indexes': indexes
        }


    

