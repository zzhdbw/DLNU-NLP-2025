# 关系抽取任务_bert初探

import transformers
from transformers import BertTokenizer, BertModel

# 加载预训练的BERT模型和tokenizer
model_name = "../pretrained_models/bert-base-chinese"

# 添加特殊符号到tokenizer

tokenizer = BertTokenizer.from_pretrained(model_name)
special_tokens_dict = {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
tokenizer.add_special_tokens(special_tokens_dict)


print("================================")
# 文本推理
text = "<e1></e1><e2></e2>"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
print("================================")


model = BertModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
outputs = model(**inputs)
print(outputs)
print("================================")
print(outputs.last_hidden_state.shape)
print("================================")
print(outputs.pooler_output.shape)
