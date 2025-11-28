# 关系抽取任务_本地大模型-关系抽取推理-单条
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "..\pretrained_models\Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

label_en_to_zh = {
    "Instrument-Agency": "工具-代理",
    "Component-Whole": "部分-整体",
    "Member-Collection": "成员-集合",
    "Cause-Effect": "原因-结果",
    "Entity-Destination": "实体-目的地",
    "Message-Topic": "消息-主题",
    "Entity-Origin": "实体-起源",
    "Content-Container": "内容-容器",
    "Other": "其他",
    "Product-Producer": "产品-生产者",
}
labels_zh = label_en_to_zh.values()
# prepare the model input
prompt = f"最常见的<e1>审计</e1>内容涉及<e2>浪费</e2>问题以及废物的回收利用。\n\n将上面文本分类成下面几个类别中的一个，{labels_zh}，考虑<e1>和<e2>标签包裹的两个实体"
print(prompt)
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

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
    "\n"
)
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
