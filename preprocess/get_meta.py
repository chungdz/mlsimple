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
parser.add_argument("--vfilep", default="valid.tsv", type=str,
                        help="valid file")
parser.add_argument("--headp", default="header.tsv", type=str,
                        help="header file")
args = parser.parse_args()

filep = os.path.join(args.dpath, args.filep)
validp = os.path.join(args.dpath, args.vfilep)
headp = os.path.join(args.dpath, args.headp)
outp = os.path.join(args.dpath, "meta_info.json")
print('load data')
header = pd.read_csv(headp, sep='\t')
df = pd.read_csv(filep, sep='\t', names=header.columns, iterator=True, chunksize=1000)
vdf = pd.read_csv(validp, sep='\t', names=header.columns, iterator=True, chunksize=1000)

# flist = [x for x in list(df.columns) if 'Feature' in x]
# id_feature = ["m:AdId", "m:OrderId", "m:CampaignId", "m:AdvertiserId", "m:ClientID", "m:TagId", "m:PublisherFullDomainHash", "m:PublisherId", "m:UserAgentNormalizedHash","m:DeviceOSHash"]
id_feature = [
                # "m:AdId", 
                "m:OrderId", 
                "m:CampaignId", 
                "m:AdvertiserId", 
                # "m:ClientID", 
                "m:TagId", 
                "m:PublisherFullDomainHash", 
                "m:PublisherId", 
                "m:UserAgentNormalizedHash",
                "m:DeviceOSHash"]

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

for chunk in tqdm(df):
    get_meta_info(chunk)

for chunk in tqdm(vdf):
    get_meta_info(chunk)

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
    "features": flist,
    "to_minus": to_minus,
    "to_div": to_div,
    "ids": id_feature,
    "dicts": idxdicts
}

json.dump(infodict, open(outp, "w"))
