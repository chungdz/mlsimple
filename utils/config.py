import json
import pickle
import numpy as np
import os

class NNConfig():
    def __init__(self, dpath='data'):

        self.hidden = 100
        self.emb_size = 10
        self.dpath = dpath
        self.meta = json.load(open(os.path.join(dpath, 'meta_info.json'), 'r'))
        self.flist = ["Feature_1_garbage1_none",
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
        self.idlist = ["m:AdId", "m:OrderId", "m:CampaignId", "m:AdvertiserId", "m:ClientID", "m:TagId", "m:PublisherFullDomainHash", "m:PublisherId", "m:UserAgentNormalizedHash","m:DeviceOSHash"]
