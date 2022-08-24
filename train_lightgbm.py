from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from nn_modules.mgtir import MGTIR
from nn_modules.noid import NoID
from utils.train_util import set_seed
from torch.utils.data import DataLoader, IterableDataset
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
# from catboost import CatBoostClassifier, Pool
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
parser.add_argument("--dpath", default="/data/yunfanhu/samples/", type=str,
                        help="root path of all data")
parser.add_argument("--filep", default="train.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="header file")
parser.add_argument("--vfilep", default="valid.tsv", type=str,
                        help="valid file")
parser.add_argument("--sfilep", default="para/fimp.tsv", type=str,
                        help="place to save importance")
args = parser.parse_args()

print('load config')
cfg = NNConfig(args.dpath)
headerp = os.path.join(args.dpath, args.headp)
trainp = os.path.join(args.dpath, args.filep)
validp = os.path.join(args.dpath, args.vfilep)
print('load data')
header = pd.read_csv(headerp, sep='\t')
tdf = pd.read_csv(trainp, sep='\t', names=header.columns)
vdf = pd.read_csv(validp, sep='\t', names=header.columns)
print('parse data')
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

for findex, fname in enumerate(cfg.flist):
    tdf[fname] = (tdf[fname] - to_sub[findex]) / to_div[findex]
    vdf[fname] = (vdf[fname] - to_sub[findex]) / to_div[findex]
                
for dindex, idname in enumerate(cfg.idlist):
    tdf[idname] = tdf[idname].apply(lambda x: dicts[dindex][str(x)])
    vdf[idname] = vdf[idname].apply(lambda x: dicts[dindex][str(x)])

x_train = tdf[cfg.flist + cfg.idlist]
y_train = tdf["m:Click"]
x_valid = vdf[cfg.flist + cfg.idlist]
y_valid = vdf["m:Click"]
id_indexes = np.arange(len(cfg.flist), len(cfg.flist + cfg.idlist)).tolist()

params = {'num_leaves': 31, 'objective': 'binary', 'metric': ['auc', 'f1']}
lgb_train = lgb.Dataset(x_train.values, y_train.values, categorical_feature=id_indexes)
lgb_eval = lgb.Dataset(x_valid.values, y_valid.values, categorical_feature=id_indexes, reference=lgb_train)

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

feature_imp = pd.DataFrame({'Value': gbm.feature_importance(), 'Feature': x_train.columns}).sort_values('Value', axis=0)
print(feature_imp)
feature_imp.to_csv(args.sfilep, sep='\t', index=None)

