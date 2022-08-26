import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import math
import json
import random
import argparse
from utils.config import NNConfig
from utils.metrics import cm
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/data/yunfanhu/gbm/", type=str,
                        help="root path of original data")
parser.add_argument("--prob_path", default="/data/yunfanhu/prob/", type=str,
                        help="root path of predicted data")
parser.add_argument("--gbm", default="/data/yunfanhu/gbm/LightGBM_predict_result.txt", type=str,
                        help="root path of predicted data")
parser.add_argument("--label_file", default="/data/yunfanhu/gbm/valid.tsv", type=str,
                        help="root path of original data")
parser.add_argument("--plots", default="plots/LightGBM.jpg", type=str,
                        help="path to save calibration plot")
args = parser.parse_args()

vpp = os.path.join(args.prob_path, 'valid_prob.tsv')

print('load data')
vp = pd.read_csv(vpp, sep='\t')
gbmp = pd.read_csv(args.gbm, names=["GBMPrediction"])
label = pd.read_csv(args.label_file, sep='\t', usecols=[2], names=['Label'])

emsemble = vp['Prediction'] + gbmp["GBMPrediction"]
sig = 1/(1 + np.exp(-emsemble.values))
label_ids = label['Label'].values
print(cm(sig, label_ids))

ctrue, cpred = calibration_curve(label_ids, sig, n_bins=500, strategy="quantile", pos_label=1)
plt.xlabel('PredictedRate')
plt.ylabel('TrueRate')
plt.gca().set_aspect('equal', 'box')
plt.scatter(cpred, ctrue, s=1)

plt.title('normal scales')
plt.plot([0, 1], [0, 1], alpha=0.3, color='yellow')

plt.savefig(args.plots)
