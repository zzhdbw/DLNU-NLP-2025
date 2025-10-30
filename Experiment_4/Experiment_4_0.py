# 关系抽取任务_数据读取
import json

# 测试集
index_to_text = {}
with open("../data/SemEval2010-Task8-Chinese/test.txt", "r", encoding="utf8") as r:
    for line in r:
        # print(line)
        index, text = line.split("\t")
        index_to_text[index] = [text.strip()]
with open(
    "../data/SemEval2010-Task8-Chinese/test_result.txt", "r", encoding="utf8"
) as r:
    for line in r:
        index, label = line.split("\t")
        index_to_text[index].append(label.strip())

train_data_list = []

for index, (text, label) in index_to_text.items():
    train_data_list.append({"text": text, "label": label})
json.dump(
    train_data_list,
    open("processed_data/test.json", "w", encoding="utf8"),
    ensure_ascii=False,
    indent=4,
)

# 训练集
index_to_text = {}
with open("../data/SemEval2010-Task8-Chinese/train.txt", "r", encoding="utf8") as r:
    for line in r:
        # print(line)
        index, text = line.split("\t")
        index_to_text[index] = [text.strip()]
with open(
    "../data/SemEval2010-Task8-Chinese/train_result.txt", "r", encoding="utf8"
) as r:
    for line in r:
        index, label = line.split("\t")
        index_to_text[index].append(label.strip())
print(index_to_text)

label_set = set()
train_data_list = []

for index, (text, label) in index_to_text.items():
    train_data_list.append({"text": text, "label": label})
    label_set.add(label)
json.dump(
    train_data_list,
    open("processed_data/train.json", "w", encoding="utf8"),
    ensure_ascii=False,
    indent=4,
)
id2label = dict([(index, label) for index, label in enumerate(label_set)])
label2id = dict([(label, index) for index, label in id2label.items()])

json.dump(id2label, open("processed_data/id2label.json", "w", encoding="utf8"))
json.dump(label2id, open("processed_data/label2id.json", "w", encoding="utf8"))
