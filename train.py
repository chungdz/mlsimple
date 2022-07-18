from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from nn_modules.mgtir import MGTIR
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

set_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data", type=str,
                        help="root path of all data")
parser.add_argument("--epoch", default=3, type=int, help="training epoch")
parser.add_argument("--batch_size", default=32, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
parser.add_argument("--save_path", default='cps', type=str, help="path to save training model parameters")
parser.add_argument("--resume_checkpoint", default=None, type=str, help='''whether to start training from scratch 
                            or load parameter saved before and continue training. For example, if start_epoch=/mnt/cifar/checkpoint-20, then model will load parameter 
                            in the path and continue the epoch of training after 20 steps''')
parser.add_argument("--filep", default="sample.tsv", type=str,
                        help="train file")
args = parser.parse_args()

print('load config')
cfg = NNConfig(args.dpath)
df = pd.read_csv(os.path.join(args.dpath, args.filep), sep='\t')
dlen = df.shape(0)
train_size = dlen * 4 // 5
trainset = df[:train_size]
validset = df[train_size:]
print('load dataset')
trainset = ClassificationTrainDS(cfg, trainset)
validset = ClassificationTrainDS(cfg, validset)

print('load model')
model = MGTIR(cfg)

print('load trainer')
training_args = TrainingArguments(
    output_dir=args.save_path,
    per_device_train_batch_size=args.batch_size,
    lr_scheduler_type='linear',
    optim="adamw_torch",
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    # logging_steps=1,
    num_train_epochs=args.epoch,
    fp16=False,
    learning_rate=args.lr,
    save_total_limit=50,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
    seed=7,
    data_seed=7
)
trainer = Trainer(model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=trainset,
    eval_dataset=validset
)

print('start training')
trainer.train(resume_from_checkpoint=args.resume_checkpoint)