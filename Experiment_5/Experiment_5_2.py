# 关系抽取任务_本地大模型-关系抽取推理-评估
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

model_name = "..\pretrained_models\Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

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


def eval_test(prompt):
    # prepare the model input

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    # print("content:", content)
    return content, extract_relations(content)


test_data_path = "processed_data/test.json"
test_data = json.load(open(test_data_path, "r", encoding="utf8"))
acc = 0
for index, line in enumerate(test_data):
    text = line["text"]
    labels_zh_str = "\n".join(labels_zh)
    prompt = f"{text}\n\n你是一名研究关系抽取的专家，你可以分辨<e1>和<e2>标签所包裹的实体之间的关系，全部关系如下：{labels_zh_str},所以两个实体之间的关系是："

    answer, prediction = eval_test(prompt)
    label = line["label"].lower()
    print(
        f"{index}, \n真标签:{label} \n预测标签:{prediction} \n大模型输出内容：{answer} \n文本:{line['text']}"
    )
    print("=" * 100)
    if prediction == label:
        acc += 1
print(f"accuracy: {acc / len(test_data)}")
