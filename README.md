# DLNU-NLP-2025
大连民族大学2025秋季学期自然语言处理课程实验部分



## 环境安装

### 创建python环境

```
conda create -n DLNU-NLP-2025 python=3.10 -y
conda activate DLNU-NLP-2025
```



### 安装torch 

参考网址：https://pytorch.org/get-started/previous-versions/

#### CUDA版本

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

#### CPU版本（无GPU情况）

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```



### 安装其他包

```sh
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```



## 数据集下载

参考网址：

[简体中文口语情感分类数据集 · 数据集](https://www.modelscope.cn/datasets/zhangzhihao/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset)
[基于SemEval2010-Task8翻译的中文关系抽取数据集 · 数据集](https://www.modelscope.cn/datasets/zhangzhihao/SemEval2010-Task8-Chinese)

执行代码以下载数据集

```bash
cd data
./download.bat
```



## 模型下载

### word2vec

参考网址：[sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5 · 模型库](https://www.modelscope.cn/models/zhangzhihao/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5)

执行代码以下载word2vec权重

```
cd pretrained_models
./download_word2vec.bat
```



### bert

参考网址：[bert-base-chinese · 模型库](https://www.modelscope.cn/models/google-bert/bert-base-chinese)

执行代码以下载bert权重

```
cd pretrained_models
./download_bert.bat
```



## 代码

### Experiment_1

> 神经网络情感分类代码

Experiment_1_1.py 数据处理代码

Experiment_1_2.py 模型训练代码

Experiment_1_3.py 模型评估代码



### Experiment_2

> word2vec情感分类代码

Experiment_2_0.py 数据处理代码

Experiment_2_1.py word2vec权重处理代码

Experiment_2_2.py 模型训练代码

Experiment_2_3.py 模型评估代码



### Experiment_3

> word2vec关系抽取代码

Experiment_3_1.py 数据处理代码

Experiment_3_2.py jieba分词测试代码

Experiment_3_3.py  word2vec权重处理代码

Experiment_3_4.py 模型训练代码

Experiment_3_5.py 模型训练代码



### Experiment_4

Experiment_4_0.py 数据处理代码

Experiment_4_1.py BERT初探

Experiment_4_2.py BERT初探

Experiment_4_3.py BERT关系分类训练

### Experiment_5

Experiment_5_0.py 大模型初探-本地

Experiment_5_1.py 本地大模型-关系抽取推理-单条

Experiment_5_2.py 关系抽取任务_本地大模型-关系抽取推理-评估

Experiment_5_3.py 关系抽取任务_在线大模型-初探

Experiment_5_4.py 关系抽取任务_在线大模型-关系抽取-评估

Experiment_5_5.py 关系抽取任务_本地大模型-关系抽取-训练