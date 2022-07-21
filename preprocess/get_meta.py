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

# flist = [x for x in list(df.columns) if 'Feature' in x]
flist = ["Feature_1_garbage1_none",
"Feature_1_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_1_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_66_garbage1_none",
"Feature_66_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_66_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_552_garbage1_none",
"Feature_552_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_552_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_594_garbage1_none",
"Feature_594_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_594_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_755_garbage1_none",
"Feature_755_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_755_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_1574_garbage1_none",
"Feature_1574_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_1574_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_1576_garbage1_none",
"Feature_1576_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_1576_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_1627_garbage1_none",
"Feature_1627_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_1627_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_1628_garbage1_none",
"Feature_1628_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_1628_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_1631_garbage1_none",
"Feature_1631_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_1631_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_25_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_25_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_97_garbage1_none",
"Feature_97_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_97_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_576_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_576_ExpectedClicks_add1thenlogthenmultiplyby1000",
"Feature_181_garbage1_none",
"Feature_181_COECUsingClicks_add1thenlogthenmultiplyby1000",
"Feature_181_ExpectedClicks_add1thenlogthenmultiplyby1000",
# "Feature_Cascade_7",
"Feature_66_garbage1_none_5",
"Feature_66_RTImpressions_add1thenlogthenmultiplyby1000_5",
"Feature_66_RTNonClicksByClicks_add1thenlogthenmultiplyby1000_5",
"Feature_7276_garbage1_none_5",
"Feature_7276_RTImpressions_add1thenlogthenmultiplyby1000_5",
"Feature_7276_RTNonClicksByClicks_add1thenlogthenmultiplyby1000_5",
"Feature_1_garbage1_none_5",
"Feature_1_RTImpressions_add1thenlogthenmultiplyby1000_5",
"Feature_1_RTNonClicksByClicks_add1thenlogthenmultiplyby1000_5",
"Feature_1194_garbage1_none_5",
"Feature_1194_RTImpressions_add1thenlogthenmultiplyby1000_5",
"Feature_1194_RTNonClicksByClicks_add1thenlogthenmultiplyby1000_5",
"Feature_163_garbage1_none_5",
"Feature_163_RTImpressions_add1thenlogthenmultiplyby1000_5",
"Feature_163_RTNonClicksByClicks_add1thenlogthenmultiplyby1000_5",
"Feature_7240_garbage1_none_5",
"Feature_7240_RTImpressions_add1thenlogthenmultiplyby1000_5",
"Feature_7240_RTNonClicksByClicks_add1thenlogthenmultiplyby1000_5"
]
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