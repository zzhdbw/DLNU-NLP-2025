# 关系抽取任务_bert初探

import transformers
from transformers import BertTokenizer, BertModel

# 加载预训练的BERT模型和tokenizer
model_name = "../pretrained_models/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertModel.from_pretrained(model_name)
print(model)
print("================================")
# 文本推理
text = "我喜欢吃苹果。"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
print("================================")
outputs = model(**inputs)
print(outputs)
print("================================")
print(outputs.last_hidden_state.shape)
print("================================")
print(outputs.pooler_output.shape)
