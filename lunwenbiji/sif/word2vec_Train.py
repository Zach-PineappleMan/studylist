import pandas as pd
from gensim.models import Word2Vec
import jieba  # 用于中文分词

# 步骤 1: 读取 Excel 文件
df = pd.read_excel('validation_data_ts_10000_原帖匹配_1121.xlsx')
texts = df['帖子文本更新'].astype(str)

# 步骤 2: 预处理文本（例如，中文分词）
sentences = [list(jieba.cut(text)) for text in texts]

# 步骤 3: 训练 Word2Vec 模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 保存模型
model.save("word2vec.model")