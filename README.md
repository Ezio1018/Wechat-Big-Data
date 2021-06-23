# **微信大数据挑战赛**

## 项目成员

- 朱国峰-186004034
- 余泽林-186003330
- 洪宇-186000837

## 目录结构

- fm.py: 模型训练
- model.py: deepfm的模型架构
- preprocess.py: 数据预处理
- run.sh:运行脚本（4个target）
- submit.py:将预测生成的npy文件，转化为csv
- preprocessed_data/: 数据
  - data.csv:训练数据集
  - test.csv:测试数据集
- result/: 预测结果
  - click_avatar.npy:点击
  - forward.npy:转发
  - like.npy:喜欢
  - read_comment.npy:读评论


##  具体思路

1. 首先对数据进行预处理，可以看到原始user_action表中只有userid,feedid和对应的action，但是训练的时候是需要feedid的属性值的，所以我们首先python preprocess.py对数据进行处理，包括填充空值，对videoSeconds进行对数转换等等。

   | userid | feedid | date_ | device | read_comment | comment | like | play | stay | click_avatar | forward | follow | favorite |
   | ------ | ------ | ----- | ------ | ------------ | ------- | ---- | ---- | ---- | ------------ | ------- | ------ | -------- |
   | 8      | 71474  | 1     | 1      | 0            | 0       | 1    | 500  | 5366 | 0            | 0       | 0      | 0        |
   | 8      | 73916  | 1     | 1      | 0            | 0       | 0    | 250  | 1533 | 0            | 0       | 0      | 0        |
   | 8      | 50282  | 1     | 1      | 0            | 0       | 0    | 750  | 1302 | 0            | 0       | 0      | 0        |
   | 8      | 11391  | 1     | 1      | 0            | 0       | 1    | 3750 | 5191 | 0            | 0       | 0      | 0        |
   | 8      | 27349  | 1     | 1      | 0            | 0       | 0    | 250  | 800  | 0            | 0       | 0      | 0        |

2. 得到的数据表如下所示

   离散变量：'userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id','device'

   连续变量：'videoplayseconds'

   标签属性：'read_comment','like','click_avatar','forward'

   | userid | feedid | device | read_comment | like | click_avatar | forward | authorid | videoplayseconds | bgm_song_id | bgm_singer_id |
   | ------ | ------ | ------ | ------------ | ---- | ------------ | ------- | -------- | ---------------- | ----------- | ------------- |
   | 8      | 71474  | 1      | 0            | 1    | 0            | 0       | 1528     | 2.484907         | 13746       | 3557          |
   | 8      | 73916  | 1      | 0            | 0    | 0            | 0       | 1442     | 2.833213         | 0           | 0             |
   | 8      | 50282  | 1      | 0            | 0    | 0            | 0       | 8648     | 3.465736         | 0           | 0             |
   | 8      | 11391  | 1      | 0            | 1    | 0            | 0       | 11976    | 1.94591          | 13097       | 5013          |

3. 由于模型只能选定一个target所以我们通过run.sh，分别执行四次，将离散变量，连续变量作为输入，而点击，转发，喜欢，读评论，分别作为模型的target进行训练。

   模型大致由两部分组成，一部分是FM，还有一部分就是DNN, 而FM又由一阶特征部分与二阶特征交叉部分组成，所以可以将整个模型拆成三部分，分别是一阶特征处理linear部分，二阶特征交叉FM以及DNN的高阶特征交叉。

   以下是前向传播图![image-20210228161135777](https://camo.githubusercontent.com/406fd9ca0fc6bf713bd97be287488b0d07d0b32f2539c7b8339ed867b57ade6e/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232383136313133353737372e706e67)

   以下是通过keras画的模型结构图

   ![DeepFM](https://camo.githubusercontent.com/2dbd28bc15f554e75d508a1d3fa4f53a834b51287d4653e710db2691c3b4b84e/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2f25453525394225424525453725383925383744656570464d2e706e67)

## 模型结果

| 提交时间            |  提交者   | 提交类型 | 得分     | 查看评论 | 点赞     | 点击头像 | 转发     |
| ------------------- | :-------: | -------- | -------- | -------- | -------- | -------- | -------- |
| 2021-06-22 23:18:59 | undefined | 初赛A榜  | 0.639841 | 0.624967 | 0.610151 | 0.701898 | 0.664294 |

一开始成绩不太理想，后来发现是嵌入层层数不够![image-20210623170002122](https://github.com/Ezio1018/Wechat-Big-Data/blob/main/%E7%BB%93%E6%9E%9C.png)

## CTR点击率模型-DeepFm

[![image-20210225180556628](https://camo.githubusercontent.com/97c425985ff1d67e80d6b7e6de1608e0f6dbc1e45dcfa10f18bb469a69c559ad/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232353138303535363632382e706e67)](https://camo.githubusercontent.com/97c425985ff1d67e80d6b7e6de1608e0f6dbc1e45dcfa10f18bb469a69c559ad/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232353138303535363632382e706e67)

- **Deep模型部分**
- **FM模型部分**
- **Sparse Feature中黄色和灰色节点代表什么意思**

###  FM

下图是FM的一个结构图，从图中大致可以看出FM Layer是由一阶特征和二阶特征Concatenate到一起在经过一个Sigmoid得到logits（结合FM的公式一起看），所以在实现的时候需要单独考虑linear部分和FM交叉特征部分。 $$ \hat{y}*{FM}(x) = w_0+\sum*{i=1}^N w_ix_i + \sum_{i=1}^N \sum_{j=i+1}^N v_i^T v_j x_ix_j $$ [![image-20210225181340313](https://camo.githubusercontent.com/dedb4c63f6d6460a4f164bd3c1db6296e6e886685ff48585528f274a9288019d/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232353138313334303331332e706e67)](https://camo.githubusercontent.com/dedb4c63f6d6460a4f164bd3c1db6296e6e886685ff48585528f274a9288019d/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232353138313334303331332e706e67)

###  Deep

Deep架构图

[![image-20210225181010107](https://camo.githubusercontent.com/6cec5898538db83081d51574beb95b3099669af8df9942553ff92f8061c3aa0e/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232353138313031303130372e706e67)](https://camo.githubusercontent.com/6cec5898538db83081d51574beb95b3099669af8df9942553ff92f8061c3aa0e/687474703a2f2f72796c756f2e6f73732d636e2d6368656e6764752e616c6979756e63732e636f6d2fe59bbee78987696d6167652d32303231303232353138313031303130372e706e67)

Deep Module是为了学习高阶的特征组合，在上图中使用用全连接的方式将Dense Embedding输入到Hidden Layer，这里面Dense Embeddings就是为了解决DNN中的参数爆炸问题，这也是推荐模型中常用的处理方法。

Embedding层的输出是将所有id类特征对应的embedding向量concat到到一起输入到DNN中。其中$v_i$表示第i个field的embedding，m是field的数量。 $$ z_1=[v_1, v_2, ..., v_m] $$ 上一层的输出作为下一层的输入，我们得到： $$ z_L=\sigma(W_{L-1} z_{L-1}+b_{L-1}) $$ 其中$\sigma$表示激活函数，$z, W, b $分别表示该层的输入、权重和偏置。

最后进入DNN部分输出使用sigmod激活函数进行激活： $$ y_{DNN}=\sigma(W^{L}a^L+b^L) $$
