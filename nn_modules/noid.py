import torch
import torch.nn as nn
import torch.nn.functional as F

class NoID(nn.Module):

    def __init__(self, cfg):
        super(NoID, self).__init__()
        
        self.hidden = cfg.hidden
        self.flen = len(cfg.flist)
        self.wd = cfg.weight_decay
        self.seq = nn.Sequential(
            nn.Linear(self.flen, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        )
        
    def predict(self, finputs, idinputs):
        return self.seq(finputs)

    def forward(self, finputs, idinputs, labels):

        logits = self.predict(finputs, idinputs)
        loss_weights = torch.clone(labels)
        loss_weights.masked_fill_(~loss_weights.bool(), self.wd)
        loss = F.binary_cross_entropy(logits.squeeze(), labels, weight=loss_weights)

        return {
            'loss': loss,
            'logits': logits
        }


    

