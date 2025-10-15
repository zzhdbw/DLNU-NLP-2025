# 文本分类任务_数据读取

import pandas as pd
import jieba
import json

if __name__ == "__main__":
    data_path = "../data/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv"

    data = pd.read_csv(data_path)
    print(data)
    data = data.to_dict("records")

    word_set = set()
    label_set = set()

    for line in data:
        label_set.add(line["label"])

        for word in jieba.lcut(line["text"]):
            word_set.add(word)

    word_set.add("<|UNK|>")
    word_set.add("<|PAD|>")

    label2id = dict([(label, index) for index, label in enumerate(label_set)])
    id2label = dict([(index, label) for index, label in enumerate(label_set)])

    word2id = dict([(word, index) for index, word in enumerate(word_set)])
    id2word = dict([(index, word) for index, word in enumerate(word_set)])

    json.dump(
        label2id,
        open("processed_data/label2id.json", "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )

    json.dump(
        id2label,
        open("processed_data/id2label.json", "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )

    json.dump(
        data,
        open("processed_data/train.json", "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )
