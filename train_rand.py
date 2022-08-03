from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from nn_modules.mgtir import MGTIR
from nn_modules.noid import NoID
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from utils.ds_rand import ClassificationTrainDS, collate_fn
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
parser.add_argument("--epoch", default=10, type=int, help="training epoch")
parser.add_argument("--batch_size", default=512, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--save_path", default='cps', type=str, help="path to save training model parameters")
parser.add_argument("--resume_checkpoint", action='store_true', help='''whether to start training from scratch 
                            or load parameter saved before and continue training. For example, if start_epoch=/mnt/cifar/checkpoint-20, then model will load parameter 
                            in the path and continue the epoch of training after 20 steps''')
parser.add_argument("--additionId", action='store_true', help='whether to add AdId and UserId')
parser.add_argument("--filep", default="sample.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="train file")
parser.add_argument("--vfilep", default="valid.tsv", type=str,
                        help="valid file")
parser.add_argument("--with_id", default=1, type=int,
                        help="default has id")
parser.add_argument("--tfilep", default=None, type=str,
                        help="test file after train")
args = parser.parse_args()

print('load config')
if args.additionId:
    print('add user id and add id')
cfg = NNConfig(args.dpath, additionId=args.additionId, no_id=(args.with_id == 0))
headerp = os.path.join(args.dpath, args.headp)
trainp = os.path.join(args.dpath, args.filep)
validp = os.path.join(args.dpath, args.vfilep)
print('load dataset')
trainset = ClassificationTrainDS(cfg, headerp, trainp)
validset = ClassificationTrainDS(cfg, headerp, validp)

print('load model')
if args.with_id == 1:
    model = MGTIR(cfg)    
else:
    model = NoID(cfg)

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
    metric_for_best_model="eval_AUC",
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

if not args.tfilep is None:
    testp = os.path.join(args.dpath, args.tfilep)
    testset = ClassificationTrainDS(cfg, headerp, testp, args.chunk_size // 4)
    trainer.evaluate(testset)