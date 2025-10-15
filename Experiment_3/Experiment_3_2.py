import jieba
import re

jieba.re_han_default = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-<>\/\|]+)", re.U)
text = (
    "如上所述的系统，在天线<e2>元件</e2>以阵列形式排列的<e1>配置</e1>中应用最为广泛。"
)
# jieba添加自定义词
jieba.add_word("<e1>")
jieba.add_word("</e1>")
jieba.add_word("<e2>")
jieba.add_word("</e2>")
text_cut = jieba.lcut(text)
print(text_cut)
