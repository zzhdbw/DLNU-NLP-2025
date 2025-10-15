# 文本分类任务_naive_infer
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Flatten
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import jieba
from sklearn.metrics import accuracy_score
import json


class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.embedding = Embedding(len(word2id) + 1, word_dim)
        self.embedding = Embedding(len(word2id), word_dim)

        self.flatten = Flatten()  # 展平向量
        self.linear = Linear(max_len * word_dim, len(label2id))

    def forward(self, input):
        output = self.embedding(input)
        output = self.flatten(output)
        output = self.linear(output)
        return output


# 定义数据集加载方式
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.texts = []  # X
        self.labels = []  # Y

        for line in data:
            self.labels.append(line["label"])
            self.texts.append(jieba.lcut(line["text"]))

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


# 定义批处理流程
def collate_fn(batch):
    batch_text = []
    batch_label = []

    for text, label in batch:
        text = [word2id.get(word, word2id["<|UNK|>"]) for word in text]
        label = label2id[label]

        if len(text) > max_len:  # 截断操作
            text = text[:max_len]
        if len(text) < max_len:  # 填充
            text.extend([word2id["<|PAD|>"]] * (max_len - len(text)))

        batch_text.append(text)
        batch_label.append(label)

    batch_text = torch.LongTensor(batch_text)
    batch_label = torch.LongTensor(batch_label)
    return batch_text.to(device), batch_label.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = 128
    word_dim = 300
    batch_size = 6
    learning_rate = 1e-4
    epoch = 100
    train_data_path = "processed_data/train.json"
    word2id_path = "processed_data/word2id.json"
    label2id_path = "processed_data/label2id.json"
    model_path = "output/best_model.pth"
    word2vec_path = "word2vec/word2vec.npy"

    ############################################################
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))

    label2id = json.load(
        open(label2id_path, "r", encoding="utf8"),
    )

    word2id = json.load(
        open(word2id_path, "r", encoding="utf8"),
    )
    train_dataset = MyDataset(train_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    model = MyModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # 评估
    total_label = []
    total_predict = []
    model.eval()
    with torch.no_grad():
        for batch_text_ids, batch_label in tqdm(train_dataloader):
            output = model(batch_text_ids)
            batch_predict = torch.argmax(output, dim=-1).tolist()

            total_label.extend(batch_label.tolist())
            total_predict.extend(batch_predict)

    # 使用sklearn计算准确率
    accuracy = accuracy_score(total_label, total_predict)
    print(f"准确率: {accuracy:.4f}")
