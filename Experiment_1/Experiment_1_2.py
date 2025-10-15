# 文本分类任务_naive_train
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Flatten
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import jieba
from sklearn.metrics import accuracy_score
import json
import os


# 定义模型结构
class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
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
    learning_rate = 1e-3
    epoch = 10
    train_data_path = "processed_data/train.json"
    word2id_path = "processed_data/word2id.json"
    label2id_path = "processed_data/label2id.json"
    output_path = "output"
    ########################################################
    word2id = json.load(open(word2id_path, "r", encoding="utf-8"))
    label2id = json.load(open(label2id_path, "r", encoding="utf-8"))

    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    train_dataset = MyDataset(train_data)

    train_dataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    model = MyModel().to(device)
    optimizer = Adam(lr=learning_rate, params=model.parameters())
    loss_fn = CrossEntropyLoss()

    best_acc = 0
    for per_epoch in range(epoch):
        # 训练
        model.train()
        for batch_text_ids, batch_label in tqdm(train_dataLoader):
            optimizer.zero_grad()  # 梯度清零
            output = model(batch_text_ids)  # 前向传播
            loss = loss_fn(output, batch_label)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降
            # print(f"{loss.item():.4}")

        # 评估
        total_label = []
        total_predict = []
        model.eval()
        with torch.no_grad():
            for batch_text_ids, batch_label in tqdm(train_dataLoader):
                output = model(batch_text_ids)
                batch_predict = torch.argmax(output, dim=-1).tolist()

                total_label.extend(batch_label.tolist())
                total_predict.extend(batch_predict)

        # 使用sklearn计算准确率
        accuracy = accuracy_score(total_label, total_predict)
        print(f"准确率: {accuracy:.4f}")

        if accuracy > best_acc:
            best_acc = accuracy
            # 只保存模型参数
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
            print(f"✅ 保存最佳模型，准确率: {accuracy:.4f}")
