import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os
import argparse
from utils.metrics import cm

parser = argparse.ArgumentParser()
parser.add_argument("--spath", default="plots/res.jpg", type=str,
                        help="save path")
parser.add_argument("--res", default="cps_small/no_id.csv", type=str,
                        help="on which result data generate the plot")
parser.add_argument("--sample", default="quantile", type=str,
                        help="quantile or uniform bin for calibration plot")
parser.add_argument("--points", default=500, type=int,
                        help="maximum number of bins as well as points in the calibration plot")
parser.add_argument("--ns", action='store_true', help='use normal scale instead of log log scale')
args = parser.parse_args()
print('load results')
m0df = pd.read_csv(args.res)
print(cm(m0df['predictions'].values, m0df['labels'].values))
print('build calibration point')
ctrue0, cpred0 = calibration_curve(m0df['labels'], 
                                m0df['predictions'], 
                                n_bins=args.points, 
                                strategy=args.sample, 
                                pos_label=1)

plt.xlabel('PredictedRate')
plt.ylabel('TrueRate')
plt.gca().set_aspect('equal', 'box')
plt.scatter(cpred0, ctrue0, s=1)
if not args.ns:
    plt.title('log-log scale ' + args.sample)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([1e-5, 1], [1e-5, 1], alpha=0.3, color='yellow')
else:
    plt.title('normal scale ' + args.sample)
    plt.plot([0, 1], [0, 1], alpha=0.3, color='yellow')

plt.savefig(args.spath)

# plt.axis('equal')
