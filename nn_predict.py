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
from utils.metrics import compute_metrics
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
parser.add_argument("--max_steps", default=300000, type=int, help="training total steps")
parser.add_argument("--save_steps", default=30000, type=int, help="training save steps")
parser.add_argument("--batch_size", default=2, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--chunk_size", default=5000, type=int, help="chunk_size * batch_size * GPU = real_batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--save_path", default='cps_20/', type=str, help="path to save prediction result")
parser.add_argument("--checkpoint", default='cps_20/m1_0821_raw/pytorch_model.bin', type=str, help="path to save training model parameters")
parser.add_argument("--filep", default="train.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="train file")
parser.add_argument("--with_id", default=1, type=int,
                        help="default has id")
args = parser.parse_args()

print('load config')
cfg = NNConfig(args.dpath, additionId=False, no_id=(args.with_id == 0))
headerp = os.path.join(args.dpath, args.headp)
tfilep = os.path.join(args.dpath, args.filep)
print('load dataset')
trainset = ClassificationTrainDS(cfg, headerp, tfilep, args.chunk_size, isTrain=False, printIndex=100)
validset = ClassificationTrainDS(cfg, headerp, tfilep, args.chunk_size, isTrain=False, printIndex=100)

print('load model')
if args.with_id == 1:
    model = MGTIR(cfg)    
else:
    model = NoID(cfg)

pretrained_model = torch.load(args.checkpoint, map_location='cpu')
print('load trained parameters', model.load_state_dict(pretrained_model, strict=False))

print('load trainer')
training_args = TrainingArguments(
    output_dir=args.save_path,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    lr_scheduler_type='linear',
    optim="adamw_torch",
    dataloader_num_workers=1,
    dataloader_pin_memory=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    metric_for_best_model="eval_ROC AUC",
    logging_steps=args.save_steps,
    eval_steps=args.save_steps,
    save_steps=args.save_steps,
    max_steps=args.max_steps,
    fp16=False,
    learning_rate=args.lr,
    save_total_limit=50,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    seed=7,
    data_seed=7,
    ignore_data_skip=True
)
trainer = Trainer(model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=trainset,
    eval_dataset=validset
)
print('predict and save')
(res, raw_res, indexes), label_ids, metrics = trainer.predict(validset)
print('raw min and max', raw_res.min(), raw_res.max(), 'total row', raw_res.shape)
final_res = np.zeros((raw_res.shape[0]))
for elem, idx in zip(raw_res, indexes):
    final_res[idx] = elem
print((final_res == 0).sum())
np.save(os.path.join(args.save_path, 'probability.npy'), final_res)
