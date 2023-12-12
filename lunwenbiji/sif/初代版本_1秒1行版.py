# -*- coding: utf-8 -*-
import argparse
import re
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


global_call_count = 0


def split_into_sentences(text):
    # 使用正则表达式定义句子的结束符号，包括常见和不常见的标点符号
    sentence_endings = r'([.。！？!\?…\n]|[\u3000]|[\u2026]|[！]{1,}|[!]{1,}| {2,})'

    # 使用正则表达式的 findall 方法来寻找所有的句子
    sentences = re.findall(sentence_endings, text)

    # 初始化一个列表来存储处理后的句子
    processed_sentences = []
    start = 0

    # 遍历每个匹配的句子结束符号
    for match in re.finditer(sentence_endings, text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence and len(sentence)>1:
            processed_sentences.append(sentence)
        start = end

    return processed_sentences

def segment_paragraphs_by_similarity(text, model, similarity_threshold=0.74):
    # 每当调用100次函数时打印一次报告
    global global_call_count
    global_call_count += 1
    if global_call_count % 10 == 0:
        print(f"已处理 {global_call_count} 条数据")

    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return text

    embeddings = model.encode(sentences, convert_to_tensor=True)
    paragraphs = []
    current_paragraph = [sentences[0]]
    current_paragraph_start = embeddings[0]

    for i in range(1, len(sentences)):
        similarity_with_paragraph_start = util.cos_sim(embeddings[i], current_paragraph_start).item()
        similarity_with_last = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        similarity_with_next = util.cos_sim(embeddings[i], embeddings[i + 1]).item() if i < len(sentences) - 1 else 0

        max_similarity = max(similarity_with_paragraph_start, similarity_with_last, similarity_with_next)

        if max_similarity == similarity_with_paragraph_start or (
                max_similarity == similarity_with_last and similarity_with_paragraph_start >= similarity_threshold):
            current_paragraph.append(sentences[i])
        else:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [sentences[i]]
            current_paragraph_start = embeddings[i]

    # 添加最后一个段落
    paragraphs.append(' '.join(current_paragraph))

    text_paragraphs = "\n".join(paragraphs)
    return text_paragraphs

# def segment_paragraphs_by_similarity(text, model, similarity_threshold=0.74):
#
#
#     sentences = split_into_sentences(text)
#
#     if len(sentences)<2:
#         return text
#     # 编码句子
#     embeddings = model.encode(sentences)
#
#     paragraphs = []
#     current_paragraph = [sentences[0]]
#     current_paragraph_start = embeddings[0]
#     i = 1
#     while i < len(sentences) - 1:
#         # 计算相似度
#         similarity_with_paragraph_start = util.cos_sim(embeddings[i], current_paragraph_start).item()
#         similarity_with_last = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
#         similarity_with_next = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
#
#         # 确定最高相似度
#         max_similarity = max(similarity_with_paragraph_start, similarity_with_last, similarity_with_next)
#
#         if max_similarity == similarity_with_paragraph_start or (
#                 max_similarity == similarity_with_last and similarity_with_paragraph_start >= similarity_threshold):
#             # 如果本句与段首相似度最高或与上一句相似度最高且与段首相似度超过阈值，不分段
#             current_paragraph.append(sentences[i])
#             i += 1
#         else:
#             # 否则，开始新段落
#             paragraphs.append(' '.join(current_paragraph))
#             current_paragraph = [sentences[i]]
#             current_paragraph_start = embeddings[i]
#             i += 1
#             if i ==len(sentences):
#                 current_paragraph.append(sentences[i])
#     # 添加最后一个段落
#     if current_paragraph:
#         paragraphs.append(' '.join(current_paragraph))
#
#
#     text_paragraphs = "\n".join(paragraphs)
#     return text_paragraphs

# 文本示例
# text = """
# 没有痘痘 皮肤光滑的小秘密
# 果然，不管春夏秋冬还是离不开烟酰胺精华液~用对烟酰胺，白成闪电不是梦!精华水质地温和，流动性很好，上脸冰冰凉凉的很好吸收，不会特别黏，很清爽。这款自然妍的烟酰胺还是高浓度的 还是院线产品 使用起来更加的放心 还省下了去美容院的小钱钱
# 请和我一样经常长痘的宝看看这支祛痘宝藏叭!!浣颜凝胶还是我的痘肌同事给我推荐的
# 刚好那段时间熬夜+上火、下巴和额头冒了好多痘就买了一支试试看
# 连着用了几天太雷了!!
# 痘痘跟打地鼠似的 瘪得贼快!小小一支出门带着也好方便
# 爆痘的时候马上就能用它急救!丝毫不慌!我比较喜欢睡前用、涂完就去睡觉睡醒红痘能瘪好多连用几天就平了
# """

# 加载模型
# model = SentenceTransformer('moka-ai/m3e-base')
# print(segment_paragraphs_by_similarity(text, model))

"""
椰子真的太神奇了，它对我们生活贡献真的是非常多，能吃能喝，没想到还能用来做天然无害的洗护品 ，泰国人流传下来的智慧，真的是把它“椰生”的价值发挥地淋漓尽致。
小C只能表示“椰椰”您太厉害啦！ 今天就给大家推荐一下这些香fufu又平价的椰子好物吧！
Bio way 椰子洗发水+护发素 适合干枯，分叉发质，洗完头发有一股淡淡的椰子味，抹点护发素头发光泽又顺滑 Sense护发油
一油多用，不仅可以涂在发梢，还能涂在皮肤干燥的地方，防止皮肤龟裂 Bio way 椰子亮白草本牙膏
茶渍、咖啡渍☕️和香烟渍统统退退退！ 每次使用绿豆大小，针对黄牙的地方刷刷刷
国联椰子舒缓凝胶 类似芦荟胶的质地，不过是椰子味儿的，除了涂在嘴唇，头发或者皮肤上有保湿作用，还有一定的提亮和淡化细纹的功效哦 能吃能用的有机椰油！
对，兄弟姐妹们，你们没听错 Agrilife，FarmDii两个牌子差距不大，FarmDii椰子味稍浓一丢丢
常规用法：当做身体乳，护发油，炒菜做饭 隐藏用法：蚊子包涂一点，可以减轻红肿和发痒哦


椰子真的太神奇了，它对我们生活贡献真的是非常多，能吃能喝，没想到还能用来做天然无害的洗护品 ，泰国人流传下来的智慧，真的是把它“椰生”的价值发挥地淋漓尽致。
小C只能表示“椰椰”您太厉害啦！ 今天就给大家推荐一下这些香fufu又平价的椰子好物吧！
Bio way 椰子洗发水+护发素 适合干枯，分叉发质，洗完头发有一股淡淡的椰子味，抹点护发素头发光泽又顺滑 Sense护发油 一油多用，不仅可以涂在发梢，还能涂在皮肤干燥的地方，防止皮肤龟裂 Bio way 椰子亮白草本牙膏
茶渍、咖啡渍☕️和香烟渍统统退退退！ 每次使用绿豆大小，针对黄牙的地方刷刷刷
国联椰子舒缓凝胶 类似芦荟胶的质地，不过是椰子味儿的，除了涂在嘴唇，头发或者皮肤上有保湿作用，还有一定的提亮和淡化细纹的功效哦 能吃能用的有机椰油！
对，兄弟姐妹们，你们没听错 Agrilife，FarmDii两个牌子差距不大，FarmDii椰子味稍浓一丢丢
常规用法：当做身体乳，护发油，炒菜做饭 隐藏用法：蚊子包涂一点，可以减轻红肿和发痒哦
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two excel files based on post_id column.')
    parser.add_argument('--file', type=str,default='test200.xlsx', help='待处理文件.xlsx')
    parser.add_argument("--column_name", default="帖子文本更新", type=str, help="列名")
    parser.add_argument('--output',type=str,default='output_test200.xlsx', help='输出结果.xlsx')
    args = parser.parse_args()

    model = SentenceTransformer('moka-ai/m3e-base')
    datapath = args.file
    df = pd.read_excel(datapath)
    print("加载完成")

    if args.column_name not in df.columns:
        raise ValueError(f"没找到 '{args.column_name}' 这一列")
    print("开始处理")
    start_time = time.time()
    df['Processed_Text'] = df[args.column_name].apply(lambda x: segment_paragraphs_by_similarity(str(x),model))
    df.to_excel(args.output, index=False)
    print("处理完成")
    print("time used {:.2f} sec".format(time.time() - start_time))