import torch
import torch.nn as nn
import torch.nn.functional as F

class MGTIR(nn.Module):

    def __init__(self, cfg):
        super(MGTIR, self).__init__()
        
        self.hidden = cfg.hidden
        self.seq = nn.Sequential(
            nn.Linear(12, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
        ) 
        
    def predict(self, data):
        return self.seq(data).squeeze()

    def forward(self, inputs, labels):

        logits = self.seq(inputs)
        loss = F.binary_cross_entropy(logits.squeeze(), labels)

        return {
            'loss': loss,
            'logits': logits
        }


    

