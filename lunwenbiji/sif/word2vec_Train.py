import pandas as pd
from gensim.models import Word2Vec
import jieba  # 用于中文分词


# 步骤 1: 读取 Excel 文件
df = pd.read_excel('test.xlsx')

# print(df.head())
# print()
# print(df.columns)

# 选择多列
columns = ['列1名', '列2名', '列12名','列9名']
texts = df[columns].astype(str).apply(lambda row: ' '.join(row), axis=1)

# 步骤 2: 预处理文本（例如，中文分词）
sentences = [list(jieba.cut(text)) for text in texts]

# 步骤 3: 训练 Word2Vec 模型
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=4, epochs=10)

# 保存模型
model.save("word2vec.model")
