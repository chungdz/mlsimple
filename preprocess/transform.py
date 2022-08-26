import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import math
import json
import random
import argparse
from utils.config import NNConfig
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/data/yunfanhu/samples_20/", type=str,
                        help="root path of original data")
parser.add_argument("--prob_path", default="/data/yunfanhu/prob/", type=str,
                        help="root path of predicted data")
parser.add_argument("--out_path", default="/data/yunfanhu/gbm/", type=str,
                        help="root path of predicted data")
parser.add_argument("--chunk_size", default=100000, type=int)
args = parser.parse_args()

filep = os.path.join(args.dpath, 'train.tsv')
validp = os.path.join(args.dpath, 'valid_5M.tsv')
headp = os.path.join(args.dpath, 'header.tsv')

tpp = os.path.join(args.prob_path, 'train_prob.tsv')
vpp = os.path.join(args.prob_path, 'valid_prob.tsv')

outp = os.path.join(args.out_path, "train.tsv")
voutp = os.path.join(args.out_path, "valid.tsv")
houtp = os.path.join(args.out_path, "header.tsv")
print('load data')
header = pd.read_csv(headp, sep='\t')
df = pd.read_csv(filep, sep='\t', names=header.columns, iterator=True, chunksize=args.chunk_size)
vdf = pd.read_csv(validp, sep='\t', names=header.columns, iterator=True, chunksize=args.chunk_size)
tp = pd.read_csv(tpp, sep='\t')
vp = pd.read_csv(vpp, sep='\t')
print('size of prediction', tp.shape, vp.shape)
print('load config')
cfg = NNConfig(args.dpath)

newh = ['Prediction', 'Target'] + list(header[['m:Click'] + cfg.flist + cfg.idlist].columns)
with open(houtp, 'w', encoding='utf8') as f:
    f.write('\t'.join(newh))
    f.write('\n')

dicts = []
to_sub = []
to_div = []

for fname in cfg.flist:
    findex = cfg.meta['all_features'].index(fname)
    to_sub.append(cfg.meta['to_minus'][findex])
    to_div.append(cfg.meta['to_div'][findex])

for idname in cfg.idlist:
    dindex = cfg.meta['all_ids'].index(idname)
    dicts.append(cfg.meta['dicts'][dindex])

vindex = 0
for chunk in tqdm(vdf, desc='scan validset', total=math.ceil(vp.shape[0] / args.chunk_size)):
    new_chunk = chunk[['m:Click'] + cfg.flist + cfg.idlist]

    cur_size = new_chunk.shape[0]
    new_chunk.insert(0, 'Target', vp['Target'][vindex: vindex + cur_size].values)
    new_chunk.insert(0, 'Prediction', vp['Prediction'][vindex: vindex + cur_size].values)
    vindex += cur_size

    for findex, fname in enumerate(cfg.flist):
        new_chunk[fname] = (new_chunk[fname] - to_sub[findex]) / to_div[findex]
    
    for dindex, idname in enumerate(cfg.idlist):
        new_chunk[idname] = new_chunk[idname].apply(lambda x: dicts[dindex].get(str(x), 0))
    
    new_chunk.to_csv(voutp, header=False, index=False, mode='a', sep='\t')

tindex = 0
for chunk in tqdm(df, desc='scan trainset', total=math.ceil(tp.shape[0] / args.chunk_size)):
    new_chunk = chunk[['m:Click'] + cfg.flist + cfg.idlist]

    cur_size = new_chunk.shape[0]
    new_chunk.insert(0, 'Target', tp['Target'][tindex: tindex + cur_size].values)
    new_chunk.insert(0, 'Prediction', tp['Prediction'][tindex: tindex + cur_size].values)
    tindex += cur_size

    for findex, fname in enumerate(cfg.flist):
        new_chunk[fname] = (new_chunk[fname] - to_sub[findex]) / to_div[findex]
    
    for dindex, idname in enumerate(cfg.idlist):
        new_chunk[idname] = new_chunk[idname].apply(lambda x: dicts[dindex].get(str(x), 0) if random.random() > 0.006 else 0)

    new_chunk.to_csv(outp, header=False, index=False, mode='a', sep='\t')
