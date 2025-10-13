# DLNU-NLP-2024
大连民族大学2024秋季学期自然语言处理课程实验部分

conda create -n DLNU-NLP-2024 python=3.10 -y
conda activate DLNU-NLP-2024

下载torch
https://pytorch.org/get-started/previous-versions/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

数据集下载
https://www.modelscope.cn/datasets/zhangzhihao/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset

下载已经训练好的word2vec权重
https://github.com/Embedding/Chinese-Word-Vectors
Baidu Encyclopedia 百度百科-Word-300d版本
文件名：sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2
解压后：sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5，放置在word2vec目录下

