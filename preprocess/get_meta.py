import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data", type=str,
                        help="root path of all data")
parser.add_argument("--filep", default="sample.tsv", type=str,
                        help="train file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="train file")
args = parser.parse_args()

filep = os.path.join(args.dpath, args.filep)
headp = os.path.join(args.dpath, args.headp)
outp = os.path.join(args.dpath, "meta_info.json")
print('load data')
header = pd.read_csv(headp, sep='\t')
df = pd.read_csv(filep, sep='\t', names=header.columns)

print(df['m:Click'].value_counts())

flist = [x for x in list(df.columns) if 'Feature' in x]
min_list = []
max_list = []
for fname in flist:
    min_list.append(int(df[fname].min()))
    max_list.append(int(df[fname].max()))

id_feature = ["m:AdId", "m:OrderId", "m:CampaignId", "m:AdvertiserId", "m:ClientID", "m:TagId", "m:PublisherFullDomainHash", "m:PublisherId", "m:UserAgentNormalizedHash","m:DeviceOSHash"]
idxdicts = []
for idname in tqdm(id_feature):
    cdict = {}
    idx = 0
    for v in df[idname]:
        if str(v) not in cdict:
            cdict[str(v)] = idx
            idx += 1
    idxdicts.append(cdict)

infodict = {
    "features": flist,
    "minlist": min_list,
    "maxlist": max_list,
    "ids": id_feature,
    "dicts": idxdicts
}

json.dump(infodict, open(outp, "w"))