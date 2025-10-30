# 关系抽取任务_bert训练
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, Flatten
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import json
import numpy as np
import os
import re

import transformers
from transformers import BertTokenizer, BertModel


class MyModel(Module):
    def __init__(self, model_name):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(
            len(tokenizer)
        )  # 调整BERT模型的词嵌入层,以匹配tokenizer的词汇表大小

        # # 冻结BERT模型参数,可以减少训练参数数量,加速训练
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.linear = Linear(768, len(label2id))

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = output.pooler_output

        output = self.linear(last_hidden_state)
        return output


# 定义数据集加载方式
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.texts = []  # X
        self.labels = []  # Y

        for line in data:
            self.labels.append(line["label"])
            self.texts.append(line["text"])

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


# 定义批处理流程
def collate_fn(batch):

    batch_label = []
    text_list = []

    for text, label in batch:
        text_list.append(text)
        batch_label.append(label2id[label])

    batch_tokenized_text = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )

    batch_label = torch.LongTensor(batch_label)
    return batch_tokenized_text.to(device), batch_label.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_len = 128
    word_dim = 300
    batch_size = 6
    learning_rate = 1e-4
    epoch = 10

    train_data_path = "processed_data/train.json"
    test_data_path = "processed_data/test.json"
    label2id_path = "processed_data/label2id.json"
    id2label_path = "processed_data/id2label.json"
    output_path = "output"
    model_name = "../pretrained_models/bert-base-chinese"
    ############################################################

    tokenizer = BertTokenizer.from_pretrained(model_name)
    special_tokens_dict = {
        "additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    label2id = json.load(
        open(label2id_path, "r", encoding="utf8"),
    )

    id2label = json.load(
        open(id2label_path, "r", encoding="utf8"),
    )

    train_data = json.load(open(train_data_path, "r", encoding="utf8"))
    test_data = json.load(open(test_data_path, "r", encoding="utf8"))

    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    model = MyModel(model_name)
    model.to(device)

    optimizer = Adam(lr=learning_rate, params=model.parameters())
    loss_fn = CrossEntropyLoss()

    best_acc = 0
    for per_epoch in range(epoch):
        # 训练
        model.train()
        train_bar = tqdm(train_dataloader)
        for batch_tokenized_text, batch_label in train_bar:
            optimizer.zero_grad()  # 梯度清零
            output = model(**batch_tokenized_text)  # 前向传播

            loss = loss_fn(output, batch_label)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 梯度下降

            train_bar.set_description(f"Epoch {per_epoch+1}/{epoch}")
            train_bar.set_postfix(loss=loss.item())

        # 评估
        total_label = []
        total_predict = []
        model.eval()
        with torch.no_grad():
            for batch_tokenized_text, batch_label in tqdm(train_dataloader):
                output = model(**batch_tokenized_text)
                batch_predict = torch.argmax(output, dim=-1).tolist()

                total_label.extend(batch_label.tolist())
                total_predict.extend(batch_predict)
        train_accuracy = accuracy_score(total_label, total_predict)
        train_f1 = f1_score(total_label, total_predict, average="macro")
        train_precision = precision_score(total_label, total_predict, average="macro")
        train_recall = recall_score(total_label, total_predict, average="macro")
        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"训练集F1: {train_f1:.4f}")
        print(f"训练集精确率: {train_precision:.4f}")
        print(f"训练集召回率: {train_recall:.4f}")

        total_label = []
        total_predict = []
        model.eval()
        with torch.no_grad():
            for batch_tokenized_text, batch_label in tqdm(test_dataloader):
                output = model(**batch_tokenized_text)
                batch_predict = torch.argmax(output, dim=-1).tolist()

                total_label.extend(batch_label.tolist())
                total_predict.extend(batch_predict)

        # 使用sklearn计算准确率
        accuracy = accuracy_score(total_label, total_predict)
        f1 = f1_score(total_label, total_predict, average="macro")
        precision = precision_score(total_label, total_predict, average="macro")
        recall = recall_score(total_label, total_predict, average="macro")
        print(f"测试集准确率: {accuracy:.4f}")
        print(f"测试集F1: {f1:.4f}")
        print(f"测试集精确率: {precision:.4f}")
        print(f"测试集召回率: {recall:.4f}")

        if accuracy > best_acc:
            best_acc = accuracy
            # 只保存模型参数
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
            print(f"保存最佳模型，准确率: {accuracy:.4f}")
