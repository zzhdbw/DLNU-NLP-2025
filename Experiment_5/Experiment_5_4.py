# 关系抽取任务_在线大模型-关系抽取-评估
# https://platform.deepseek.com/api_keys
import os
from openai import OpenAI

# $env:DEEPSEEK_API_KEY = "sk-XXXX"
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm


label_en_to_zh = {
    "Instrument-Agency".lower(): "工具-代理",
    "Component-Whole".lower(): "部分-整体",
    "Member-Collection".lower(): "成员-集合",
    "Cause-Effect".lower(): "因果关系",
    "Entity-Destination".lower(): "实体-目的地",
    "Message-Topic".lower(): "消息-主题",
    "Entity-Origin".lower(): "实体-起源",
    "Content-Container".lower(): "内容-容器",
    "Other".lower(): "其他",
    "Product-Producer".lower(): "产品-生产者",
}
label_zh_to_en = dict([(v, k) for (k, v) in label_en_to_zh.items()])
labels_zh = label_en_to_zh.values()
labels_en = label_en_to_zh.keys()


def extract_relations(text):
    text = text.strip()
    text = text.lower()

    if text in label_zh_to_en:
        return label_zh_to_en[text]

    for label in label_zh_to_en.keys():
        if label in text:
            return label_zh_to_en[label]

    return "other"


def get_response(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    print(response.choices[0].message.content)

    content = response.choices[0].message.content
    return content, extract_relations(content)


test_data_path = "processed_data/test.json"
test_data = json.load(open(test_data_path, "r", encoding="utf8"))
acc = 0
for index, line in enumerate(test_data):
    text = line["text"]
    labels_zh_str = "\n".join(labels_zh)
    prompt = f"{text}\n\n你是一名研究关系抽取的专家，你可以分辨<e1>和<e2>标签所包裹的实体之间的关系，全部关系如下：{labels_zh_str},所以两个实体之间的关系是："

    answer, prediction = get_response(prompt)
    label = line["label"].lower()
    print(
        f"{index}, \n真标签:{label} \n预测标签:{prediction} \n大模型输出内容：{answer} \n文本:{line['text']}"
    )
    print("=" * 100)
    if prediction == label:
        acc += 1
print(f"accuracy: {acc / len(test_data)}")
