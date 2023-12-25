# -*- coding: utf-8 -*-

import argparse
import time
from pprint import pprint

import pandas as pd
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline(
    task=Tasks.document_segmentation,
    model='./nlp_bert_document-segmentation_chinese-base')

def segment_documents(text):
    result = p(documents=text)

    return result[OutputKeys.TEXT]

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Merge two excel files based on post_id column.')
#     parser.add_argument('--file', type=str, default='test200.xlsx', help='待处理文件.xlsx')
#     parser.add_argument("--column_name", default="帖子文本更新", type=str, help="列名")
#     parser.add_argument('--output', type=str, default=str(time.time())+'output_test.xlsx', help='输出结果.xlsx')
#     args = parser.parse_args()
#     # 导入模型
#     pprint("load model")
#
#     # 读取数据
#     pprint("read file")
#     datapath = args.file
#     df = pd.read_excel(datapath)
#     if args.column_name not in df.columns:
#         raise ValueError(f"没找到 '{args.column_name}' 这一列")
#
#     pprint("开始处理")
#     start_time = time.time()
#     # 方案1 按照文本长度灵活分割
#     # df['Processed_Text1'] = df[args.column_name].apply(lambda x: segment_paragraphs_by_itself(str(x), model))
#     # 方案2 按照相似度阈值进行切割
#     # df['Processed_Text033'] = df[args.column_name].apply(lambda x: segment_paragraphs_by_similarity(str(x), model,similarity_threshold=0.33))
#     # 方案3
#     df['Processed_Text_new'] = df[args.column_name].apply(lambda x: segment_documents(str(x)))
#
#     df.to_excel(args.output, index=False)
#     pprint("处理完成")
#     pprint("time used {:.2f} sec".format(time.time() - start_time))

text1 = """
没有痘痘 皮肤光滑的小秘密
果然，不管春夏秋冬还是离不开烟酰胺精华液~用对烟酰胺，白成闪电不是梦!精华水质地温和，流动性很好，上脸冰冰凉凉的很好吸收，不会特别黏，很清爽。这款自然妍的烟酰胺还是高浓度的 还是院线产品 使用起来更加的放心 还省下了去美容院的小钱钱。
请和我一样经常长痘的宝看看这支祛痘宝藏叭！浣颜凝胶还是我的痘肌同事给我推荐的 刚好那段时间熬夜+上火、下巴和额头冒了好多痘就买了一支试试看。
连着用了几天太雷了！ 痘痘跟打地鼠似的 瘪得贼快!小小一支出门带着也好方便  爆痘的时候马上就能用它急救!丝毫不慌!我比较喜欢睡前用、涂完就去睡觉睡醒红痘能瘪好多连用几天就平了。
"""

text2= """
除了护手霜，近期用比较多就是各种安/瓶精华～终于把手头上的都用掉了！都是小红书好物体验之前抽中的好物奖/品～其次是冬天必/备的暖宝宝，贴在后背真的全身暖呼呼的…
# 沉浸式消耗[话题]
# 笔记灵感[话题]
# 沉浸式补货[话题]
# 沉浸式[话题]
# 解压日常[话题]
# 解压[话题]
# 空瓶记[话题]
# 护肤品空瓶打卡[话题]
# 日常消耗[话题]
# 沉浸式[话题]
"""

print(segment_documents(text1))
print("------------------")
print(segment_documents(text2))
