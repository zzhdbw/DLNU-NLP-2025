# 文本分类任务_处理word2vec
import numpy as np
from tqdm import tqdm
import json

word2vec_path = "../pretrained_models/word2vec/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"

# 读取原始word2vec权重
with open(word2vec_path, "r", encoding="utf8") as r:
    temp = r.readline()

    word_len, dim_len = temp.split(" ")
    print(f"word_len:{word_len}, dim_len:{dim_len}")
    index = 0
    vec_list = []
    word_list = []
    word_set = set()
    for line in tqdm(r):
        line_list = line.strip().split(" ")
        word = line_list[0]
        # 防止存在重复词
        if word in word_set:
            continue
        else:
            word_set.add(word)
        vec = line_list[1:]
        vec = np.array(vec).astype(np.float32)

        word_list.append(word)
        vec_list.append(vec)

    # print(word_list)
    # print(vec_list.shape)
    word_list.append("<|UNK|>")
    word_list.append("<|PAD|>")

    vec_list.append(np.zeros(int(dim_len)).astype(np.float32))
    vec_list.append(np.random.random((int(dim_len),)).astype(np.float32))

    word_list.append("<e1>")
    word_list.append("</e1>")
    word_list.append("<e2>")
    word_list.append("</e2>")
    vec_list.append(np.random.random((int(dim_len),)).astype(np.float32))
    vec_list.append(np.random.random((int(dim_len),)).astype(np.float32))
    vec_list.append(np.random.random((int(dim_len),)).astype(np.float32))
    vec_list.append(np.random.random((int(dim_len),)).astype(np.float32))

    vec_list = np.stack(vec_list)


word2id = dict([(word, index) for index, word in enumerate(word_list)])
id2word = dict([(int(index), word) for index, word in enumerate(word_list)])

# word2id
json.dump(
    word2id,
    open("processed_data/word2id.json", "w", encoding="utf8"),
    indent=4,
    ensure_ascii=False,
)
# 保存id2word
json.dump(
    id2word,
    open("processed_data/id2word.json", "w", encoding="utf8"),
    indent=4,
    ensure_ascii=False,
)
# 保存向量
np.save("word2vec/word2vec.npy", vec_list)
