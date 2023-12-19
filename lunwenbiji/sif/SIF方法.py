# -*- coding: utf-8 -*-

import argparse
import re
import time
import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from pprint import pprint

global_call_count = 0
# 定义一个函数，用于准备数据
def prepare_data(list_of_seqs):
    # 计算每个序列的长度
    lengths = [len(s) for s in list_of_seqs]
    # 计算样本数
    n_samples = len(list_of_seqs)
    # 找到最长序列的长度
    maxlen = np.max(lengths)

    # 初始化一个全为零的数组，用于存放单词索引
    x = np.zeros((n_samples, maxlen)).astype('int32')
    # 初始化一个全为零的掩码数组
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')

    # 填充上述两个数组
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.

    # 确保掩码数组的类型为 float32
    x_mask = np.asarray(x_mask, dtype='float32')

    # 返回单词索引数组和掩码数组
    return x, x_mask


# 定义一个函数，用于将句子转换为单词索引的序列
def getSeq(s, words_map):
    """
    单独的句子和字典
    逐个将单词在字典中的只
    """
    idx = []
    # 遍历句子中的每个单词
    for w in s:
        # 如果单词在字典中，添加其索引
        if w in words_map:
            idx.append(words_map[w])

        else:

            idx.append(len(words_map) - 1)
    return idx

# 定义一个函数，将句子列表转换为单词索引数组
def sentences2idx(sentences, words):
    # 遍历句子列表
    seq1 = []
    for s in sentences:
        seq1.append(getSeq(s, words))
    # 将输入改编为 [x1，x2,x3,...]
    # 将长度统一
    x1, m1 = prepare_data(seq1)
    return x1, m1

def getWordWeight(word_frequency, a=1e-3):
    # 如果参数a小于或等于0，将其设为1.0，意味着不进行加权
    if a <= 0:
        a = 1.0
    # 初始化一个字典来存储每个单词的权重, {词1:权重1,词2:权重2}
    word2weight = {}
    # 初始化单词总频率计数为0
    N = 0
    # 计算总频率并初始化权重
    for w, f in word_frequency.items():
        # 将每个单词的频率存储在word2weight字典中
        word2weight[w] = f
        # 累加所有单词的频率
        N += f
    # 计算每个单词的权重
    for key, value in word2weight.items():
        # 使用特定的公式计算权重
        # 这个权重是通过平滑参数 a 和单词的相对频率（该单词频率与总频率之比）来确定的。
        # 高频率 假装该词出现 100次 在总共5000词的文本
        word2weight[key] = a / (a + value / N)

    # word2weight ['str']是'str'这个词的权重
    # 返回每个单词的权重字典
    return word2weight

def cut(string, stop_words=None):
    if not stop_words:
        stop_words = set()

    def token(txt):
        return ''.join(re.findall(r'[\u4e00-\u9fa5]+', txt))

    return [w for w in jieba.lcut(token(string)) if len(w.strip()) > 0 and w not in stop_words]

# 将单词的权重映射到它们在词汇表中的索引上 {1:权重，2：权重}
def getWeight(words, word2weight):
    # 初始化一个字典来存储每个单词索引的权重
    weight4ind = {}
    # 遍历单词及其索引
    for word, ind in words.items():
        # 如果单词在权重字典中，则使用其计算得到的权重
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            # 如果单词不在权重字典中，则默认其权重为1.0，  权重越大说明越冷门，内涵越多
            weight4ind[ind] = 1.0
    # weight4ind[i] is the weight for the i-th word
    return weight4ind

def seq2weight(seq, mask, weight4ind):
    """
    W表示Weighted，意为使用预估计的参数给句中的每个词向量赋予权重
    :param seq:
    :param mask:
    :param weight4ind:
    :return:
    """
    # 初始化一个与输入序列相同形状的全零权重矩阵
    weight = np.zeros(seq.shape).astype('float32')
    # 遍历序列的每一行
    for i in range(seq.shape[0]):
        # 遍历序列的每一列
        for j in range(seq.shape[1]):
            # 如果mask在该位置为正，且seq在该位置的值非负
            if mask[i, j] > 0 and seq[i, j] >= 0:
                # 根据seq在该位置的值，从weight4ind中获取权重，并赋值给weight矩阵相应位置
                weight[i, j] = weight4ind[seq[i, j]]

    # 确保weight是一个numpy数组，并且数据类型为float32
    weight = np.asarray(weight, dtype='float32')

    # 返回加权后的矩阵
    return weight


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_
def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    # 获取样本数量，即句子的数量
    n_samples = x.shape[0]
    # 初始化一个全零矩阵，用于存储每个句子的加权平均向量
    emb = np.zeros((n_samples, We.shape[1]))
    # 遍历每个句子
    for i in range(n_samples):
        # 计算第i个句子的加权平均词向量
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    # 返回句子的加权平均向量
    return emb

def SIF_embedding(We, x, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    # 计算加权平均词向量
    emb = get_weighted_average(We, x, w)
    # 如果参数params.rmpc大于0，移除句子嵌入向量在其第一个主成分上的投影
    if params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    # 返回句子嵌入向量
    return emb

class params(object):
    # 类的构造函数，当创建类的实例时自动调用
    def __init__(self):
        self.LW = 1e-5 # 初始化参数LW
        self.LC = 1e-5 # 初始化参数LC
        self.eta = 0.05  # 初始化参数eta

    # 重写__str__方法，当尝试将对象转换为字符串时调用
    def __str__(self):
        # 创建一个包含类属性及其值的元组
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        # 使用map函数将元组中的所有元素转换为字符串
        t = map(str, t)
        # 使用join方法将字符串列表连接成一个单独的字符串，并返回
        return ' '.join(t)

class SIFModel:
    def __init__(self, word2vec_model, stop_words, weightpara=1e-3):
        """
        :param word2vec_model: word2vec模型
        :param stop_words: 停用词
        :param weightpara:
        """
        # 原版本gensim3.x,而目前升级了，gensim4.0了

        # 停用词列表
        self.stop_words = stop_words
        # 词向量模型
        self.word2vec_model = word2vec_model
        # 从Word2Vec模型中获取所有单词列表
        # 旧代码 words = word2vec_model.wv.index2word
        words = word2vec_model.wv.index_to_key  # 新代码
        # 获取Word2Vec模型中每个单词的向量
        self.word_vectors = word2vec_model.wv.vectors
        # 计算并存储每个单词的频率
        # 旧代码：word_frequency = {w: v.count for w, v in word2vec_model.wv.vocab.items()}
        word_frequency = {word: word2vec_model.wv.get_vecattr(word, 'count') for word in words}  # 新代码
        # 生成一个映射，将每个单词映射到一个唯一的索引
        self.word_index_map = {w: n for n, w in enumerate(words)}
        # 计算每个单词的权重
        word2weight = getWordWeight(word_frequency, weightpara)
        # 将单词的权重映射到其索引
        self.weight4ind = getWeight(self.word_index_map, word2weight)

    def sentence_similarity(self, s1, s2):
        # get SIF embedding
        if type(s1) == list:
            s1_embdding = self.sentence2vec(s1)[0]
        else:
            s1_embdding = self.sentence2vec([s1])[0]
        if type(s2) == list:
            s2_embdding = self.sentence2vec(s2)[0]
        else:
            s2_embdding = self.sentence2vec([s2])[0]
        return self.similarity(s1_embdding, s2_embdding)

    @staticmethod
    def similarity(emb1, emb2):
        """
        句子Embedding用途
        得到句子的表示后，可以通过向量的内积计算句子相似度：
        计算两个向量余弦相似度
        :param emb1:
        :param emb2:
        :return:
        """
        inn = (emb1 * emb2).sum()
        emb1norm = np.sqrt((emb1 * emb1).sum())
        emb2norm = np.sqrt((emb2 * emb2).sum())
        scores = inn / emb1norm / emb2norm
        return scores

    def sentence2vec(self, sentences):
        """
        计算句子向量
        :param sentences:
        :return:
        """
        sens = [cut(s, self.stop_words) for s in sentences]
        x, m = sentences2idx(sens, self.word_index_map)
        w = seq2weight(x, m, self.weight4ind)
        p = params()
        p.rmpc = 0 if len(sentences) <= 1 else 1
        return SIF_embedding(self.word_vectors, x, w, p)


def split_into_sentences(text):
    # 使用正则表达式定义句子的结束符号，忽略数字中的点
    sentence_endings = r'(?<!\d)[.。！？!\?…\n](?!\d)|[\u3000]|[\u2026]|[！]{1,}|[!]{1,}| {2,}'

    sentences = re.findall(sentence_endings, text)
    processed_sentences = []
    start = 0

    for match in re.finditer(sentence_endings, text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence and len(sentence) > 1:
            processed_sentences.append(sentence)
        start = end

    return processed_sentences

def segment_paragraphs_by_similarity(text, model, similarity_threshold=0.74):
    global global_call_count
    global_call_count += 1
    if global_call_count % 100 == 0:
        print(f"已处理 {global_call_count} 条数据")

    sentences = split_into_sentences(text)
    # print("---------")
    # print(sentences)
    if len(sentences) < 2:
        return text

    paragraphs = []
    current_paragraph = [sentences[0]]

    for i in range(1, len(sentences)):

        # 计算当前句子与当前段落中最后一个句子的相似度
        sim_with_last = model.sentence_similarity(sentences[i], current_paragraph[-1])
        pprint(sim_with_last)
        # 如果相似度高于阈值，则将句子添加到当前段落
        if sim_with_last >= similarity_threshold:
            current_paragraph.append(sentences[i])
        else:
            # 相似度低于阈值，当前段落结束，开始新段落
            paragraphs.append('\n'.join(current_paragraph))
            current_paragraph = [sentences[i]]

    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(''.join(current_paragraph))

    # 将段落列表转换为字符串
    paragraphs_str = '\n'.join(paragraphs)

    return paragraphs_str
def segment_paragraphs_by_itself(text, model):
    # 计算相邻句子间的相似度
    global global_call_count
    global_call_count += 1
    if global_call_count % 100 == 0:
        print(f"已处理 {global_call_count} 条数据")

    sentences = split_into_sentences(text)

    if len(sentences) <2:
        return text
    else:
        num_paragraphs = int(len(sentences)/5+1)
        # print("num_paragraphs:")
        # print(num_paragraphs)
        #
        # print("len:")
        # print(len(sentences))
    similarities = [model.sentence_similarity(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
    # 找到相似度最低的分段点
    lowest_similarity_indexes = sorted(range(len(similarities)), key=lambda i: similarities[i])[:num_paragraphs - 1]
    # 按照这些点进行分段
    paragraphs = []
    start = 0
    for index in sorted(lowest_similarity_indexes):
        paragraphs.append(''.join(sentences[start:index+1]))
        start = index + 1
    paragraphs.append(''.join(sentences[start:]))

    return '\n'.join(paragraphs)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Merge two excel files based on post_id column.')
#     parser.add_argument('--file', type=str, default='test200.xlsx', help='待处理文件.xlsx')
#     parser.add_argument("--column_name", default="帖子文本更新", type=str, help="列名")
#     parser.add_argument('--output', type=str, default=str(time.time())+'output_test.xlsx', help='输出结果.xlsx')
#     args = parser.parse_args()
#
#     # 导入模型
#     pprint("load sif model")
#     model_path = Word2Vec.load("./best/my_word2vec.model")
#     stop_words = None
#     model = SIFModel(model_path, stop_words)
#
#     # 转化为向量
#     pprint("read file")
#     datapath = args.file
#     df = pd.read_excel(datapath)
#     if args.column_name not in df.columns:
#         raise ValueError(f"没找到 '{args.column_name}' 这一列")
#
#     pprint("开始处理")
#     start_time = time.time()
#     # 方案1 按照文本长度灵活分割
#     df['Processed_Text1'] = df[args.column_name].apply(lambda x: segment_paragraphs_by_itself(str(x), model))
#     # 方案2 按照相似度阈值进行切割
#     df['Processed_Text2'] = df[args.column_name].apply(lambda x: segment_paragraphs_by_similarity(str(x), model,similarity_threshold=0.52))
#
#     df.to_excel(args.output, index=False)
#     pprint("处理完成")
#     pprint("time used {:.2f} sec".format(time.time() - start_time))
#
model_path = Word2Vec.load("./best/my_word2vec.model")
stop_words = None
model = SIFModel(model_path, stop_words)
data = """
没有痘痘 皮肤光滑的小秘密
果然，不管春夏秋冬还是离不开烟酰胺精华液~用对烟酰胺，白成闪电不是梦!精华水质地温和，流动性很好，上脸冰冰凉凉的很好吸收，不会特别黏，很清爽。这款自然妍的烟酰胺还是高浓度的 还是院线产品 使用起来更加的放心 还省下了去美容院的小钱钱
请和我一样经常长痘的宝看看这支祛痘宝藏叭!!浣颜凝胶还是我的痘肌同事给我推荐的
刚好那段时间熬夜+上火、下巴和额头冒了好多痘就买了一支试试看
连着用了几天太雷了!!
痘痘跟打地鼠似的 瘪得贼快!小小一支出门带着也好方便
爆痘的时候马上就能用它急救!丝毫不慌!我比较喜欢睡前用、涂完就去睡觉睡醒红痘能瘪好多连用几天就平了
"""

print(segment_paragraphs_by_similarity(data, model,similarity_threshold=0.4))



