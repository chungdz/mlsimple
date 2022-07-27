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
outp = os.path.join(args.dpath, "no_id_train.tsv")
voutp = os.path.join(args.dpath, "no_id_valid.tsv")
print('load data')
header = pd.read_csv(headp, sep='\t')
df = pd.read_csv(filep, sep='\t', names=header.columns, iterator=True, chunksize=args.chunk_size)
vdf = pd.read_csv(validp, sep='\t', names=header.columns, iterator=True, chunksize=args.chunk_size)

flist = [x for x in list(header.columns) if 'Feature' in x]
id_feature = ["m:AdId", "m:OrderId", "m:CampaignId", "m:AdvertiserId", "m:ClientID", "m:TagId", "m:PublisherFullDomainHash", "m:PublisherId", "m:UserAgentNormalizedHash","m:DeviceOSHash"]

for chunk in tqdm(df):
    new_chunk = chunk.drop(columns=id_feature)
    new_chunk.to_csv(outp, header=False, index=False, mode='a', sep='\t')

for chunk in tqdm(vdf):
    new_chunk = chunk.drop(columns=id_feature)
    new_chunk.to_csv(voutp, header=False, index=False, mode='a', sep='\t')
