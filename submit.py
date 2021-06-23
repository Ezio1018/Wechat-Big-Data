import pandas as pd
import numpy as np

read_comment=np.load("wechat/read_comment.npy")
forward=np.load("wechat/forward.npy")
click_avatar=np.load("wechat/click_avatar.npy")
like=np.load("wechat/like.npy")
test = pd.read_csv("D:/Code/wechat/preprocessed_data/test.csv")

all_target=['read_comment','like','click_avatar','forward']
test['read_comment']=read_comment
test['like']=like
test['click_avatar']=click_avatar
test['forward']=forward
print(test.head())
test.to_csv("result.csv",index=None,float_format='%.6f')

