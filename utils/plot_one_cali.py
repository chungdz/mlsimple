import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--spath", default="plots/res.jpg", type=str,
                        help="save path")
parser.add_argument("--res", default="cps_small/no_id.csv", type=str,
                        help="root path of all data")
parser.add_argument("--points", default=500, type=int,
                        help="default has id")
args = parser.parse_args()
print('load results')
m0df = pd.read_csv(args.res)
print('build calibration point')
ctrue0, cpred0 = calibration_curve(m0df['labels'], 
                                m0df['predictions'], 
                                n_bins=args.points, 
                                strategy="uniform", 
                                pos_label=1)

plt.xlabel('PredictedRate')
plt.ylabel('TrueRate')
plt.title('log-log scale')
plt.xscale('log')
plt.yscale('log')
plt.axis('equal')
plt.scatter(cpred0, ctrue0, s=1)
plt.savefig(args.spath)

