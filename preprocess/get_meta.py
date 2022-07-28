import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import argparse
from utils.config import NNConfig

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data", type=str,
                        help="root path of all data")
parser.add_argument("--filep", default="sample.tsv", type=str,
                        help="train file")
parser.add_argument("--vfilep", default="valid.tsv", type=str,
                        help="valid file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="header file")
parser.add_argument("--chunk_size", default=1000, type=int)
args = parser.parse_args()

filep = os.path.join(args.dpath, args.filep)
validp = os.path.join(args.dpath, args.vfilep)
headp = os.path.join(args.dpath, args.headp)
outp = os.path.join(args.dpath, "meta_info.json")
print('load data')
header = pd.read_csv(headp, sep='\t')
df = pd.read_csv(filep, sep='\t', names=header.columns, iterator=True, chunksize=args.chunk_size)
vdf = pd.read_csv(validp, sep='\t', names=header.columns, iterator=True, chunksize=args.chunk_size)

flist = [x for x in list(header.columns) if 'Feature' in x]
ugelist = ['uge{}'.format(l) for l in range(32)]
agelist = ['age{}'.format(l) for l in range(32)]
flist = flist + ugelist + agelist
id_feature = ["m:OrderId", "m:CampaignId", "m:AdvertiserId", "m:TagId", "m:PublisherFullDomainHash", "m:PublisherId", "m:UserAgentNormalizedHash","m:DeviceOSHash"]

ilen = len(id_feature)
flen = len(flist)
idxdicts = idxdicts = [{} for _ in range(ilen)]
idxrecord = [0] * ilen
min_list = []
max_list = []

def get_meta_info(chunk):

    global max_list, min_list

    cmin_list = []
    cmax_list = []
    for fname in flist:
        cmin_list.append(int(chunk[fname].min()))
        cmax_list.append(int(chunk[fname].max()))
    
    llen = len(max_list)
    if llen == 0:
        min_list = cmin_list
        max_list = cmax_list
    else:
        assert(llen == flen)
        for i in range(llen):
            if cmin_list[i] < min_list[i]:
                min_list[i] = cmin_list[i]
            if cmax_list[i] > max_list[i]:
                max_list[i] = cmax_list[i]
                
    for j in range(ilen):
        idname = id_feature[j]
        for v in chunk[idname]:
            if v not in idxdicts[j]:
                idxdicts[j][v] = idxrecord[j]
                idxrecord[j] += 1

total_row = 0
positive_row = 0
for chunk in tqdm(df):
    total_row += chunk.shape[0]
    positive_row += chunk['m:Click'].sum()
    get_meta_info(chunk)

print('train', total_row, positive_row, total_row - positive_row, 
                        (total_row - positive_row) / positive_row)

total_row = 0
positive_row = 0
for chunk in tqdm(vdf):
    total_row += chunk.shape[0]
    positive_row += chunk['m:Click'].sum()
    get_meta_info(chunk)

print('validation', total_row, positive_row, total_row - positive_row, 
                        (total_row - positive_row) / positive_row)

to_minus = []
to_div = []
for i in range(flen):
    if min_list[i] == max_list[i] or (min_list[i] == 0 and max_list[i] == 1):
        to_minus.append(0)
        to_div.append(1)
    else:
        to_minus.append(min_list[i])
        to_div.append(max_list[i] - min_list[i])

infodict = {
    "all_features": flist,
    "to_minus": to_minus,
    "to_div": to_div,
    "all_ids": id_feature,
    "dicts": idxdicts
}

json.dump(infodict, open(outp, "w"))
