# 用于实现实体链接任务

## 环境以及包需求:
- Anaconda + Pycharm + NVIDIA GeForce RTX 3080
- Python 3.8    
- torch 1.13.0+cu117
- tqdm

## 测试集上最高的acc与对应的超参设置:
- 这是一开始在margin0.2上训练的模型上测试的最好效果和训练时间，选了候选实体中最高得分和真实链接实体得分之间计算损失
- 对应log1.txt,model1.pth
![img.png](img.png)
- 后面把margin改成0.1，然后计算损失方法换成计算真是实体和每一个候选实体之间的差异值
- 对应log3.txt,model3.pth
![img_1.png](img_1.png)
总感觉损失计算还有点问题
- margin=0.1
- 超参设置：
  - emb_size=300
  - hid_size=150
  - epochs=500
  - clip_grad=1
  - lr=1e-5
  - momentum=0.9
  - b_size=4
  - margin=0.2 max_margin损失函数差额值
- 运行时间： 1h

## 项目结构：
```
--dataset/
  -- 存放数据集文件
--saved_model/
  -- model.pth 训练好的模型
  -- log.txt 训练日志文件
-- run.sh
  运行脚本
--main.py
  -- 主函数入口，包括超参、数据加载，模型加载，模型训练，模型评估以及日志记录
  这里需要注意对于模型特征的打包处理，注意真实实体索引的记录等
  然后就是损失的计算，这里我一开始使用的是将每个待链接提及的真实实体和候选实体中得分最高的实体计算损失值，
  以0.2的差额值为目标
--model.py
  -- 模型类的实现，包括网络层的定义, 前向传播过程
  这里比较需要注意的就是候选实体的表征向量的提取，以及和句子，文档表征计算相似度时的维度问题
--prepare_data.py
  -- 数据预处理，包括数据集的加载，模型所需特征的提取等
  需要注意真实实体在候选实体中的索引，应该从0开始，而不是从1开始，否则会出错。
  这里对于文档中不存在于word_vec的token我是直接删除的
```

## 项目运行:
1. 训练: `bash run.sh`
2. 测试: `需要再run.sh中的添加LOAD参数加载模型, 然后执行bash run.sh`# Hierarchical-attention-for-entity-linking
