
import datetime
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchkeras import summary, Model

from sklearn.metrics import auc, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')


target = ["read_comment", "like", "click_avatar", "forward"]#预测
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
dense_features = ['videoplayseconds', ]
data = pd.read_csv('wechat/data/wechat_algo_data1/user_action.csv')
feed = pd.read_csv('wechat/data/wechat_algo_data1/feed_info.csv')
feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
data = data.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                    on='feedid')
test = pd.read_csv('wechat/data/wechat_algo_data1/test_a.csv')
test = test.merge(feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']], how='left',
                    on='feedid')

# 1.fill nan dense_feature and do simple Transformation for dense features
data[dense_features] = data[dense_features].fillna(0, )
test[dense_features] = test[dense_features].fillna(0, )

data[dense_features] = np.log(data[dense_features] + 1.0)
test[dense_features] = np.log(test[dense_features] + 1.0)
del data['date_']


print('data.shape', data.shape)
print('data.columns', data.columns.tolist())

data.to_csv('wechat/preprocessed_data/data.csv', index=0)
test.to_csv('wechat/preprocessed_data/test.csv', index=0)