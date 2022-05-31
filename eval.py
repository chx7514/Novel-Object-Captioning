import numpy as np
import time
import os
import pickle
import opts
import pdb
import json
from misc import utils
import pandas as pd

df=pd.DataFrame()
noc_object = ['bottle','bus',  'couch', 'microwave', 'pizza', 'racket', 'suitcase', 'zebra']
# dataset= ['/root/Desktop/pj3/annotations_DCC_clean/captions_split_set_%s_val_test_novel2014.json'%item for item in noc_object]

pred_file='save/preds_test.pkl'
# prediction
with open(pred_file, 'rb') as f:
    pred = pickle.load(f)

# for idx in range(len(dataset)):
#     lang_stats = utils.language_eval(dataset[idx], pred, noc_object[idx], 'test', '')
#     df[str(noc_object[idx])]=['%.1f'%(v*100) for k, v in lang_stats.items()]
#     df.index = [k for k, v in lang_stats.items()]

# print(df)

annFile = '/root/Desktop/pj3/annotations_DCC_clean/alltest.json'
lang_stats = utils.language_eval(annFile, pred, str(1), 'test', '')