import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--spath", default="plots/res.jpg", type=str,
                        help="save path")
parser.add_argument("--m1", default="cps_small/res.csv", type=str,
                        help="root path of all data")
parser.add_argument("--m0", default="cps_small/no_id.csv", type=str,
                        help="root path of all data")
args = parser.parse_args()

m1df = pd.read_csv(args.m1)
m0df = pd.read_csv(args.m0)

plt.xlabel('PredictedRate')
plt.ylabel('TrueRate')
plt.title('log-log scale')
plt.xscale('log')
plt.yscale('log')
plt.axis('equal')
plt.scatter(m0df['predictions'], m0df['labels'], label='Baseline', s=1)
plt.scatter(m1df['predictions'], m1df['labels'], label='M1', s=1)
plt.legend()
plt.savefig(args.spath)

