# 基于sif的长文本分段

任务目的：
输入：纯纯的长文本
输出：标题们；分段后的句子们

> 输入:是个长句子？长句子很有趣。短语才是重点。改变世界。多喝热水。
> 输出：是个长句子？长句子很有趣。\n 短语才是重点。\n 改变世界。\n 多喝热水。

已有方案：
先分成小短句，然后根据相似度直接进行合并，然后代码见 **初始方案.py** （利用了sentence_transformers的预训练模型m3e-base）
1. 分句
2. 进行编码
3. 根据相似度进行判断

可选方案：任务主题一致和语义相似度的相似度还是有差距的，所以

方案1是再进行训练，然后增加分段为几段（设置参数，使得分段结果再按照一定依据进行再次合并），等等。

方案2是换另一个方法，不仅仅是是文本相似度，可以考虑更多（比如主谓宾这些结构化的东西，例如一个句子缺少主语，就直接和前一句是一段的，或者是感叹词啥的，不单独为一段）等等。

下面是参考项目

## 参考项目：

<https://github.com/DeqianBai/Automatically-extract-news-person-speech/tree/master>
 
项目步骤：

1. 分句
2. 句子依存分析
   - 构建依存树
   - 查找说话的实体和内容
     - 查找说话实体：状中结构查找实体
     - 获取实体说的话
3. 根据句子相似度进行判断
4. 低于某个阈值则判定为结束

## 改编后：
将任意长文本进行分段。

1. 分句（不变）
2. 进行编码（SIF方法）
3. 利用规则去划分：

```python
def segment_paragraphs_by_similarity(text, model, similarity_threshold=0.6):
    global global_call_count
    global_call_count += 1
    if global_call_count % 100 == 0:
        print(f"已处理 {global_call_count} 条数据")

    sentences = split_into_sentences_v2(text)
    # print("---------")
    # print(sentences)
    len_s = len(sentences)
    if len_s < 2:
        # print("ok")
        return ''
    # 标题
    # current_paragraph = [sentences[0]]
    paragraphs = []
    # paragraphs.append(' '.join(current_paragraph))
    # 正式内容
    current_paragraph = [sentences[0]]
    # 每个都会被看一遍
    for i in range(1,len_s):
        # 计算当前句子与当前段落中段末的相似度
        sim_with_last = model.sentence_similarity(sentences[i], current_paragraph[-1])

        # 计算当前句子与当前段落中段首的相似度
        sim_with_first = model.sentence_similarity(sentences[i], current_paragraph[0])
        # 计算当前句子与当下一句的相似性
        if i<len_s-1:
            sim_with_next = model.sentence_similarity(sentences[i], sentences[i+1])
        else:
            sim_with_next = 0

        print(sim_with_first,sim_with_last,sim_with_next)

        # 判断逻辑:与段首或上一句的一致性最好
        if (sim_with_first>sim_with_last and sim_with_first>sim_with_next) or sim_with_last>sim_with_next :
            current_paragraph.append(sentences[i])
        # 如果与下一段最相似,则开启新的一段
        elif sim_with_last >= similarity_threshold:

            current_paragraph.append(sentences[i])
        else:
            # 当前句子与下一句更相似，结束当前段落，开始新段落
            paragraphs.append(''.join(current_paragraph))
            current_paragraph = [sentences[i]]

    paragraphs.append(''.join(current_paragraph))
    # print(paragraphs[-1])
    paragraphs_str = '\n'.join(paragraphs)

    return paragraphs_str
```

6. 进行输出


经过实验，好消息是有点效果，坏消息是效果不符合期望，且6万条数据也没有很好的提升。

