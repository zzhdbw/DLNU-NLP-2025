# 文本分类任务_数据读取

import pandas as pd

data = pd.read_csv(
    "data/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv"
)
print(data)
