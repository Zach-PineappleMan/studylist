import pandas as pd
from gensim.models import Word2Vec
import jieba  # 用于中文分词


# 清洗空值和只有数字的行
def is_valid_text(text):
    return text.strip() != '' and not text.isdigit()

# 步骤 1: 读取 Excel 文件
df = pd.read_excel('test.xlsx')

# 选择多列
columns = ['数据1', '数据2', '数据3']
texts = df[columns].astype(str)

texts['combined'] = texts.apply(lambda row: ' '.join(row), axis=1)
texts = texts['combined'].apply(is_valid_text)

# 步骤 2: 预处理文本（例如，中文分词）
sentences = [list(jieba.cut(text)) for text in texts]

# 步骤 3: 训练 Word2Vec 模型
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=4, epochs=10)

# 保存模型
model.save("word2vec.model")
