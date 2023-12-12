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
# 测试文本 """

# 加载模型
# model = SentenceTransformer('moka-ai/m3e-base')
# print(segment_paragraphs_by_similarity(text, model))


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
