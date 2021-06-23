import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import numpy as np  
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat,get_feature_names,DenseFeat
# import tensorflow as tf
import sys
def main():
    stage = sys.argv[1]
    print('Stage: %s'%stage)
    all_target=['read_comment','like','click_avatar','forward']
    data = pd.read_csv("D:\Code\wechat\preprocessed_data\data.csv")
    test_data = pd.read_csv("D:/Code/wechat/preprocessed_data/test.csv")
    dense_features = ['videoplayseconds']
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id','device']
    target = [stage]
    del data['play']
    del data['stay']
    del data['comment']
    del data['follow']
    del data['favorite']
    for del_target in all_target:
        if(del_target!=target[0]):
            print("delete")
            print(del_target)
            del data[del_target]

    
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=16)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    train, test = data,test_data
    train_model_input = {name:train[name].values for name in feature_names}
    test_model_input = {name:test[name].values for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adagrad", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=512, epochs=2, verbose=1, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=512*4)
    # print("test MSE", round(mean_squared_error(
    #     test[target].values, pred_ans), 4))
    print(pred_ans)
    # print(type(pred_ans))
    np.save(target[0]+".npy",pred_ans)

if __name__ == "__main__":
    main()