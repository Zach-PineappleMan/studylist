# -*- coding: utf-8 -*-

import pprint
import re
import jieba
import numpy as np
from sklearn.decomposition import TruncatedSVD

# from gensim.models import Word2Vec

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
            # 否则，添加一个特定的索引（通常表示未知单词）
            # 所以word_map应该有个占位符：见下面的例子
            # words_map = {
            #     "细节1需要从零开始":0,"这": 1, "是": 2, "一": 3, "个": 4, "测": 5, "试": 6,
            #     "我": 7, "们": 8, "在": 9, "学": 10, "习": 11, "自": 12,
            #     "然": 13, "语": 14, "言": 15, "处": 16, "理": 17,
            #     "模": 18, "型": 19, "很": 20, "有": 21,"细节2需要占位符来进行未见过的词": 22}
            idx.append(len(words_map) - 1)
    return idx

# 定义一个函数，将句子列表转换为单词索引数组
def sentences2idx(sentences, words):
    """
    句子们 和 字典 作为输入
    句子们：[句子1，句子2，句子3...]
    字典：{'字1':0,
          '字2':1,
          '字3':2...}
    getSeq()

    :param sentences: 一个句子列表
    :param words: 字典，{'str':index}  单词['str']是单词'str'的索引
    :return: x1, m1. x1[i, :] 是句子i中的单词索引, m1[i,:] 是句子i的掩码（0表示该位置没有单词）
    """
    # 遍历句子列表
    seq1 = []
    for s in sentences:
        seq1.append(getSeq(s, words))
    # 将输入改编为 [x1，x2,x3,...]
    # 将长度统一
    x1, m1 = prepare_data(seq1)
    return x1, m1

# # 捏造一些中文数据和对应的单词索引字典
# sentences = [
#     "这是一个测试",
#     "我们在学习自然语言处理",
#     "这个模型很有趣"
# ]
#
# words_map = {
#     "细节1需要从零开始":0,"这": 1, "是": 2, "一": 3, "个": 4, "测": 5, "试": 6,
#     "我": 7, "们": 8, "在": 9, "学": 10, "习": 11, "自": 12,
#     "然": 13, "语": 14, "言": 15, "处": 16, "理": 17,
#     "模": 18, "型": 19, "很": 20, "有": 21,"细节2需要占位符来进行未见过的词": 22
# }
#
#
# sens = [cut(s, None) for s in sentences]
# x, m = sentences2idx(sentences, words_map)
# pprint.pprint(x)
# print("---------------")
# pprint.pprint(m)

"""
x是从字典来的

x1
array([[ 1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0],
       [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17],
       [ 1,  4, 18, 19, 20, 21, 22,  0,  0,  0,  0]])
---------------
m是描述句子本来长度

m1
array([[1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.]], dtype=float32)
"""

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
    """
    分词
    :param stop_words:
    :param string:
    :return:
    """
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
        # 停用词列表
        self.stop_words = stop_words
        # 词向量模型
        self.word2vec_model = word2vec_model
        # 从Word2Vec模型中获取所有单词列表
        words = word2vec_model.wv.index2word
        # 获取Word2Vec模型中每个单词的向量
        self.word_vectors = word2vec_model.wv.vectors
        # 计算并存储每个单词的频率
        word_frequency = {w: v.count for w, v in word2vec_model.wv.vocab.items()}
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

    def is_next(self, s1, s2, threshold=0.5):
        """
        :param s1:
        :param s2:
        :param threshold:
        :return: 一个布尔值：True/False
        """
        return self.sentence_similarity(s1, s2) >= threshold

# words = {
#     "apple": 0,    # 苹果的索引是 0
#     "orange": 1,   # 橙子的索引是 1
#     # 香蕉的索引是 2
#     "grape": 3,    # 葡萄的索引是 3
#     "melon": 4,     # 瓜的索引是 4
#     "left":5
# }
# word2weight = {'apple': 0.0011997600479904018,
#  'banana': 0.009900990099009901,
#  'grape': 0.0234375,
#  'melon': 0.03846153846153846,
#  'orange': 0.5454545454545454}
#
# pprint.pprint(getWeight(words,word2weight))
#  {0: 0.0011997600479904018,
#  1: 0.5454545454545454,
#  3: 0.0234375,
#  4: 0.03846153846153846,
#  5: 1.0}




# word_frequency_example = {
#     "apple": 999,    # 苹果出现了999次
#     "orange": 1,    # 橙子出现了1次
#     "banana": 120,   # 香蕉出现了120次
#     "grape": 50,     # 葡萄出现了50次
#     "melon": 30      # 瓜出现了30次
# }

# pprint.pprint(getWordWeight(word_frequency_example))
"""
{'apple': 0.0011997600479904018,
 'banana': 0.009900990099009901,
 'grape': 0.0234375,
 'melon': 0.03846153846153846,
 'orange': 0.5454545454545454}
"""






