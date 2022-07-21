from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from nn_modules.mgtir import MGTIR
from nn_modules.noid import NoID
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from utils.ds import ClassificationTrainDS, collate_fn
from utils.config import NNConfig
from utils.metrics import compute_metrics
import torch
import os
import argparse
import gc
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

set_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data", type=str,
                        help="root path of all data")
parser.add_argument("--batch_size", default=32, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--load_from", default=None, type=str, help='''checkpoint path''')
parser.add_argument("--filep", default="sample.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="train file")
parser.add_argument("--with_id", default=1, type=int,
                        help="default has id")
args = parser.parse_args()

print('load config')
cfg = NNConfig(args.dpath)
header = pd.read_csv(os.path.join(args.dpath, args.headp), sep='\t')
df = pd.read_csv(os.path.join(args.dpath, args.filep), sep='\t', names=header.columns)

print('load dataset')
dataset = ClassificationTrainDS(cfg, df)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True, collate_fn=collate_fn)

print('load model')
if args.with_id == 1:
    model = MGTIR(cfg)
else:
    model = NoID(cfg)

pretrained_model = torch.load(args.load_from, map_location='cpu')
print('load trained parameters', model.load_state_dict(pretrained_model, strict=False))

model.eval()  
batch_res = []
labels = []
with torch.no_grad():
    for data in tqdm(data_loader, total=len(data_loader), desc="predict"):
        res = model.predict(data['finputs'], data['idinputs'])
        batch_res.extend(res.numpy().tolist())
        labels.extend(data['labels'].numpy().tolist())

print({
        "accuracy": accuracy_score(labels, batch_res),
        "f1_score": f1_score(labels, batch_res),
        "AUC": roc_auc_score(labels, batch_res)
    })

df = pd.DataFrame(zip(batch_res, labels), columns=['predict', 'label'])
sorted_df = df.sort_values(by='predict')
sorted_df.to_csv('results/predict.csv', index=None)




