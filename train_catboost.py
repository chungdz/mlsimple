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
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class CatBoostEvalMetricAUC(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # the larger metric value the better
        return True

    def evaluate(self, approxes, target, weight=None):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)
        return roc_auc_score(target, preds), len(preds) if weight is None else sum(weight)

set_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data", type=str,
                        help="root path of all data")
parser.add_argument("--epoch", default=30, type=int, help="training epoch")
parser.add_argument("--batch_size", default=8, type=int, help="training batch size used in Pytorch DataLoader")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--save_path", default='cps', type=str, help="path to save training model parameters")
parser.add_argument("--resume_checkpoint", default=None, type=str, help='''whether to start training from scratch 
                            or load parameter saved before and continue training. For example, if start_epoch=/mnt/cifar/checkpoint-20, then model will load parameter 
                            in the path and continue the epoch of training after 20 steps''')
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

for index, idname in enumerate(cfg.idlist):
    cdict = cfg.meta['dicts'][index]
    df[idname] = df[idname].map(cdict)

dlen = df.shape[0]
train_size = dlen * 4 // 5
flist = [x for x in list(df.columns) if 'Feature' in x]
wholeset = df[flist + cfg.idlist]
id_indexes = np.arange(len(cfg.flist), len(cfg.flist + cfg.idlist)).tolist()
wholelabel = df['m:Click']

trainset = wholeset[:train_size]
trainlabel = wholelabel[:train_size]
validset = wholeset[train_size:]
validlabel = wholelabel[train_size:]

# train_pool = Pool(trainset, 
#                   trainlabel, 
#                   cat_features=cfg.idlist,
#                   feature_names=list(trainset.columns))
# test_pool = Pool(validset,
#                 validlabel,
#                 cat_features=cfg.idlist,
#                 feature_names=list(validset.columns))

# model = CatBoostClassifier(iterations=200, 
#                         depth=6,
#                         border_count=254,
#                         learning_rate=1, 
#                         loss_function='Logloss',
#                         random_strength=1,
#                         one_hot_max_size=8,
#                         l2_leaf_reg=3)

# model.fit(train_pool, early_stopping_rounds=5, eval_set=test_pool, use_best_model=True, log_cout=open('result/output.txt', 'w'))
# model.save_model('para/catboost.cbm')

params = {'num_leaves': 31, 'objective': 'binary', 'metric': ['auc', 'f1']}
lgb_train = lgb.Dataset(trainset.values, trainlabel.values, categorical_feature=id_indexes)
lgb_eval = lgb.Dataset(validset.values, validlabel.values, categorical_feature=id_indexes, reference=lgb_train)

results = {}
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=200,
            valid_sets=lgb_eval,
            categorical_feature=id_indexes,
            evals_result=results,
            early_stopping_rounds=20)

gbm.save_model('para/lgbm.txt')
print(results)