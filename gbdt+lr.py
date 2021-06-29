import pandas as pd 
import numpy as np
import lightgbm as lgb 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 读取数据
def preProcess():
    path = 'data/'
    df_train = pd.read_csv("wechat\preprocessed_data\data.csv")
    df_test = pd.read_csv("wechat/preprocessed_data/test.csv")
    df_test['read_comment'] = -1
    del df_train['play']
    del df_train['stay']
    del df_train['comment']
    del df_train['follow']
    del df_train['favorite']
    all_target=['read_comment','like','click_avatar','forward']
    del df_train['like']
    del df_train['click_avatar']
    del df_train['forward']

    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv('data.csv', index = False)
    return data



def gbdt_lr_predict(data, category_feature, continuous_feature): 
    # 类别特征one-hot编码
    
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    train = data[data['read_comment'] != -1]
    target = train.pop('read_comment')
    test = data[data['read_comment'] == -1]
    test.drop(['read_comment'], axis = 1, inplace = True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2020)

    n_estimators = 32       
    num_leaves = 64
    # 开始训练gbdt，使用100课树，每课树64个叶节点
    model = lgb.LGBMRegressor(objective='binary',
                            subsample= 0.8,
                            min_child_weight= 0.5,
                            colsample_bytree= 0.7,
                            num_leaves=num_leaves,
                            learning_rate=0.05,
                            n_estimators=n_estimators,
                            random_state = 2020)
    model.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)],
            eval_names = ['train', 'val'],
            eval_metric = 'binary_logloss',
            verbose=0)
    # 得到每一条训练数据落在了每棵树的哪个叶子结点上
    # pred_leaf = True 表示返回每棵树的叶节点序号
    gbdt_feats_train = model.predict(train, pred_leaf = True)
    
    # 打印结果的 shape：
    print(gbdt_feats_train.shape)
    # 打印前5个数据：
    print(gbdt_feats_train[:5])
    
    # 同样要获取测试集的叶节点索引
    gbdt_feats_test = model.predict(test, pred_leaf = True)
    
    # 将 32 课树的叶节点序号构造成 DataFrame，方便后续进行 one-hot
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(n_estimators)]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)
    train_len = df_train_gbdt_feats.shape[0]
    data = pd.concat([df_train_gbdt_feats, df_test_gbdt_feats])

    # 对每棵树的叶节点序号进行 one-hot
    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    train = data[: train_len]
    test = data[train_len:]
    
    # 划分 LR 训练集、验证集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)
    
    # 开始训练lr
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    # 对测试集预测
    y_pred = lr.predict_proba(test)[:, 1]

if __name__ == '__main__':
    data = preProcess()
    dense_features = ['videoplayseconds']
    sparse_features = ['userid', 'authorid','feedid' ,'bgm_song_id', 'bgm_singer_id','device']
    gbdt_lr_predict(data, sparse_features, dense_features)