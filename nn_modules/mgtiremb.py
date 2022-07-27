import torch
import torch.nn as nn
import torch.nn.functional as F

class MGTIREmb(nn.Module):

    def __init__(self, cfg):
        super(MGTIREmb, self).__init__()
        
        self.hidden = cfg.hidden
        self.idlen = len(cfg.idlist)
        self.flen = len(cfg.flist)
        self.wd = cfg.weight_decay
        self.ualen = cfg.uaemb * 2
        self.seq = nn.Sequential(
            nn.Linear(self.flen, self.hidden),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.idlen * cfg.emb_size, self.hidden // 2),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Linear(self.hidden + self.hidden, 1),
            nn.Sigmoid())
        self.seq4 = nn.Sequential(
            nn.Linear(self.ualen, self.hidden // 2),
            nn.ReLU()
        )
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
        concated = torch.cat([self.seq(finputs[:-self.ualen]), 
                                self.seq2(embt), 
                                self.seq4(finputs[-self.ualen:])], dim=-1)
        return self.seq3(concated)

    def forward(self, finputs, idinputs, labels):

        logits = self.predict(finputs, idinputs)

        loss_weights = torch.clone(labels)
        loss_weights.masked_fill_(~loss_weights.bool(), self.wd)
        loss = F.binary_cross_entropy(logits.squeeze(), labels, weight=loss_weights)

        return {
            'loss': loss,
            'logits': logits
        }


    

