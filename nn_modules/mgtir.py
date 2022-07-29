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
        self.uaemb = cfg.uaemb
        self.seq = nn.Sequential(
            nn.Linear(self.flen, self.hidden),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Linear(self.idlen * cfg.emb_size, self.hidden // 2),
            nn.ReLU()
        )
        self.seq4 = nn.Sequential(
            nn.Linear(self.uaemb * 2, self.hidden // 4),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Linear(self.hidden + self.hidden // 2 + self.hidden // 4, 1),
            nn.Sigmoid())
        
        selected = []
        for idname in cfg.idlist:
            dindex = cfg.meta['all_ids'].index(idname)
            selected.append(cfg.meta['dicts'][dindex])
        self.embLayer = nn.ModuleList([nn.Embedding(len(d), cfg.emb_size) for d in selected])

        self.unull_emb = nn.Parameter(torch.randn(self.uaemb), requires_grad=True)
        self.anull_emb = nn.Parameter(torch.randn(self.uaemb), requires_grad=True)
        
    def predict(self, finputs, idinputs, masks):

        embs = []
        for i in range(self.idlen):
            embs.append(self.embLayer[i](idinputs[:, i]))
        embt = torch.cat(embs, dim=-1)

        uemb = finputs[:, -self.uaemb * 2: -self.uaemb]
        aemb = finputs[:, -self.uaemb:]
        
        batch_size = uemb.size(0)
        unull_emb = self.unull_emb.repeat(batch_size, 1)
        anull_emb = self.anull_emb.repeat(batch_size, 1)

        umask = masks[:, 0].unsqueeze(-1)
        amask = masks[:, 1].unsqueeze(-1)
        
        uemb = uemb * (1 - umask) + unull_emb * umask
        aemb = aemb * (1 - amask) + anull_emb * amask

        uaemb = torch.cat([uemb, aemb], dim=-1)

        concated = torch.cat([self.seq(finputs[:, :-self.uaemb * 2]), self.seq2(embt), self.seq4(uaemb)], dim=-1)
        return self.seq3(concated)

    def forward(self, finputs, idinputs, masks, labels):

        logits = self.predict(finputs, idinputs, masks)

        loss_weights = torch.clone(labels)
        loss_weights.masked_fill_(~loss_weights.bool(), self.wd)
        loss = F.binary_cross_entropy(logits.squeeze(), labels, weight=loss_weights)

        return {
            'loss': loss,
            'logits': logits
        }

