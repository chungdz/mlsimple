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
parser.add_argument("--dpath", default="/data/yunfanhu/samples", type=str,
                        help="root path of all data")
parser.add_argument("--max_steps", default=305, type=int, help="training total steps")
parser.add_argument("--save_steps", default=305, type=int, help="training save steps")
parser.add_argument("--batch_size", default=2, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--chunk_size", default=2048, type=int, help="chunk_size * batch_size * GPU = real_batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--save_path", default='cps_samples', type=str, help="path to save training model parameters")
parser.add_argument("--resume_checkpoint", action='store_true', help='''whether to start training from scratch 
                            or load parameter saved before and continue training. For example, if start_epoch=/mnt/cifar/checkpoint-20, then model will load parameter 
                            in the path and continue the epoch of training after 20 steps''')
# parser.add_argument("--additionId", action='store_true', help='whether to add AdId and UserId')
parser.add_argument("--filep", default="train_5M.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="header file")
parser.add_argument("--vfilep", default="valid_1M.tsv", type=str,
                        help="valid file")
parser.add_argument("--with_id", default=1, type=int,
                        help="if set to 0, no id features will be used")
parser.add_argument("--tfilep", default=None, type=str,
                        help="test file after train")
parser.add_argument("--points", default=500, type=int,
                        help="maximum number of bins as well as points in the calibration plot")
parser.add_argument("--plots", default="plots/cali_train.jpg", type=str,
                        help="path to save calibration plot")
args = parser.parse_args()

print('load config')
cfg = NNConfig(args.dpath, additionId=False, no_id=(args.with_id == 0))
headerp = os.path.join(args.dpath, args.headp)
trainp = os.path.join(args.dpath, args.filep)
validp = os.path.join(args.dpath, args.vfilep)
print('load dataset')
trainset = ClassificationTrainDS(cfg, headerp, trainp, args.chunk_size, isTrain=True)
validset = ClassificationTrainDS(cfg, headerp, validp, args.chunk_size, isTrain=False)

print('load model')
if args.with_id == 1:
    model = MGTIR(cfg)    
else:
    model = NoID(cfg)

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

print('start training')
trainer.train(resume_from_checkpoint=args.resume_checkpoint)
print('predict and plot')
(res, raw_res, indexes), label_ids, metrics = trainer.predict(validset)
print('raw min and max', raw_res.min(), raw_res.max())
ctrue, cpred = calibration_curve(label_ids, res.flatten(), n_bins=args.points, strategy="quantile", pos_label=1)
df = pd.DataFrame({'labels': label_ids, 'predictions': res.flatten()})
df.to_csv(os.path.join(args.save_path, 'res.csv'), index=None)

plt.xlabel('PredictedRate')
plt.ylabel('TrueRate')
plt.title('log-log scale')
# plt.axis('equal')
plt.gca().set_aspect('equal', 'box')
plt.xscale('log')
plt.yscale('log')
plt.scatter(cpred, ctrue, s=1)
plt.plot([1e-5, 1], [1e-5, 1], alpha=0.3, color='yellow')
plt.savefig(args.plots)
