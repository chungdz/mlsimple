import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import argparse
from utils.config import NNConfig
import collections

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data", type=str,
                        help="root path of all data")
args = parser.parse_args()

filep = os.path.join(args.dpath, "freq_info.json")
outp = os.path.join(args.dpath, "meta_info.json")
print('load data')
cdict = json.load(open(filep, 'r'))

def getIndexDict(curi, masked=10):
    curd = cdict['freq'][curi]
    curname = cdict['all_ids'][curi]
    print(curname, len(curd))
    newd = {'<UNK>': 0}
    newidx = 1

    for k, v in curd.items():
        if v > masked:
            newd[k] = newidx
            newidx += 1
        else:
            newd[k] = 0
    
    return newd

ndlist = [getIndexDict(i) for i in range(len(cdict['freq']))]
metadict = {
    "all_features": cdict["all_features"],
    "to_minus": cdict["to_minus"],
    "to_div": cdict["to_div"],
    "all_ids": cdict["all_ids"],
    "dicts": ndlist,
    "total_count": cdict["total_count"]
}

json.dump(metadict, outp)