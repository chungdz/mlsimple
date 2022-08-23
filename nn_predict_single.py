from tqdm import tqdm, trange
import numpy as np
import json
import pandas as pd
from nn_modules.mgtir import MGTIR
from nn_modules.noid import NoID
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from utils.ds import ClassificationTrainDS, collate_fn
from utils.config import NNConfig
from utils.metrics import compute_metrics, cm
import torch
import os
import argparse
import gc
import torch.nn.functional as F
from datasets import load_dataset, load_metric
from transformers import Trainer, TrainingArguments
import math
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

set_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/data/yunfanhu/samples_20/", type=str,
                        help="root path of all data")
parser.add_argument("--batch_size", default=2, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--chunk_size", default=20000, type=int, help="chunk_size * batch_size * GPU = real_batch_size")
parser.add_argument("--save_path", default='cps_20/', type=str, help="path to save prediction result")
parser.add_argument("--checkpoint", default='cps_20/m1_0821_raw/pytorch_model.bin', type=str, help="path to save training model parameters")
parser.add_argument("--filep", default="train.tsv", type=str,
                        help="train file")
parser.add_argument("--resp", default="prob.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="train file")
parser.add_argument("--with_id", default=1, type=int,
                        help="default has id")
parser.add_argument("--total_len", default=543886254, type=int,
                        help="total length of row")
args = parser.parse_args()

print('load config')
cfg = NNConfig(args.dpath, additionId=False, no_id=(args.with_id == 0))
headerp = os.path.join(args.dpath, args.headp)
tfilep = os.path.join(args.dpath, args.filep)
resp = os.path.join(args.save_path, args.resp)
print('load dataset')
validset = ClassificationTrainDS(cfg, headerp, tfilep, args.chunk_size, isTrain=False)
dl = DataLoader(validset,
                collate_fn=collate_fn,
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True, 
                prefetch_factor=20)

print('load model')
if args.with_id == 1:
    model = MGTIR(cfg)    
else:
    model = NoID(cfg)

pretrained_model = torch.load(args.checkpoint, map_location='cpu')
print('load trained parameters', model.load_state_dict(pretrained_model, strict=False))
# model.to(0)

data_iter = tqdm(enumerate(dl), 
                desc="validate", 
                total=args.total_len // (args.batch_size * args.chunk_size))
preds, truths, imp_ids = list(), list(), list()

with torch.no_grad():
    
    for i, data in data_iter:

        imp_ids += data['indexes'].numpy().tolist()
        truths += data['labels'].long().numpy().tolist()
        
        # 1. Forward
        predict = model(data['finputs'], 
                    data['idinputs'], 
                    data['labels'], 
                    data['indexes'])
        
        preds += predict['raw'].squeeze().cpu().numpy().tolist()

total_row = len(preds)
print('raw min and max', min(preds), max(preds), 'total row', total_row)
final_res = np.zeros((total_row))
target = np.zeros((total_row))
for elem, idx, clabel in tqdm(zip(preds, imp_ids, truths), total=total_row, desc='reorganize'):
    final_res[idx] = elem
    if clabel == 0:
        target[idx] = -15 - elem
    else:
        target[idx] = 15 - elem

print('number of blank', (final_res == 0).sum())
print('save')
with open(resp, 'w', encoding='utf-8') as f:
    f.write('Prediction' + '\t' + 'Target\n')
    for i in range(total_row):
        f.write(final_res[i])
        f.write('\t')
        f.write(target[i])
        f.write('\n')

# print('calculate metrics')
# print(cm(1/(1 + np.exp(-final_res)), label_np))
