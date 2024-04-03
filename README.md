# 记录一下自己学了哪些项目及学习进度

| 项目名称             | 链接                                                         | 学习进度(%) | 开始时间     | 是否还在学 |学习笔记链接|
|------------------|------------------------------------------------------------|---------|----------|-------|-------|
| paddleNLP/NER | https://paddlenlp.readthedocs.io/zh/latest/data_prepare/overview.html | 30 | 之前|  否 | 已经能自主训练实体、关系的抽取，其他相关功能和范式都有所了解|
|uie_pytorch|https://github.com/HUSTAI/uie_pytorch/tree/main|0|20240329|是|暂无|
|XLM-RoBERTa|https://modelscope.cn/models/iic/nlp_xlmr_named-entity-recognition_viet-ecommerce-title|15|20240401|否|进行研究，主要为跨语种+NER，目前到了瓶颈，不太明白用from modelscope.pipelines如何转到huggingface的范式上，实在不行,基于XLM-RoBERTa-NER进行微调|
|generalized-fairness-metrics|https://github.com/amazon-science/generalized-fairness-metrics/|10|20240403|是|使用亚马逊的方案，尝试完成英文任务，再进行其他语种的任务|
| SIF              | https://github.com/PrincetonML/SIF/tree/master             | 100    | 20231205 | 是     |  内容已完成，任务效果不理想，仍需努力见：studylist/lunwenbiji/sif/ |
| meta-transformer | https://github.com/invictus717/MetaTransformer/tree/master | 1       | 20231205 | 是     |   |
| OneLLM           | https://github.com/csuhan/OneLLM/tree/main                 |  0      |  20231211| 是     |   |
| LongChainKBQA     | https://github.com/wp931120/LongChainKBQA/tree/main        |    0   | 20231221  | 是    | |
|  Ai-edu   |  https://microsoft.github.io/ai-edu/                                 |  0   | 20231221|  是     |   |
|   sentence-transformers(之m3e)   |   https://github.com/wangyuxinwhy/uniem    |     0    | 20231228   | 是   |   |
| Intent Learning（意图识别）  |    👇      | 0        |   20240112   |     否|  |
| 意图识别  |  https://github.com/thuiar/TEXTOIR  | 0        |   20240123   | 是|  |
|  lang2sql   |     https://github.com/RamiKrispin/lang2sql  |   100      | 20240115    |   是    | 通过prompt对数据库提问，做了一些提升和适应，去除了csv，直接访问数据库了 见studylist/projects/sql_twice/|
|   DyFSS        |https://github.com/q086/DyFSS/tree/master                |  0    | 20240116   |   否  | |
|   SQL-GPT    | https://github.com/CL-lau/SQL-GPT     |  100    | 20240118   |   是  | |
|   snowChat   | https://github.com/kaarthik108/snowChat             |  100  | 20240118   |   是  | |
|  Natural Language to SQL   |[Natural Language to SQL ](https://medium.com/dataherald/fine-tuning-gpt-3-5-turbo-for-natural-language-to-sql-4445c1d37f7c)             |  0    | 20240118   |   是  | |
|   dataherald        |https://github.com/Dataherald/dataherald               |  0    | 20240118   |   是  | |
|   sqlcoder        |https://github.com/defog-ai/sqlcoder        |  0    | 20240118   |   是  | |
|   SQL Genius   |[SQL Genius](https://sqlgenius.app/?continueFlag=061684c79f7db7318d778e88d5acfc6e)       |  0    | 20230118   |   是  | |
|   sqlcoder        |https://github.com/defog-ai/sqlcoder        |  0    | 20240118   |   是  | |
|   ChatSQL        |https://github.com/cubenlp/ChatSQL  |  0    | 20240118   |   是  | |
|   talktosql        |http://github.com/woniesong92/talktosql  |  0    | 20240118   |   是  | |
|   sqlchat        |https://github.com/sqlchat/sqlchat|  0    | 20240118   |   是  | |
|   nllb        |  https://github.com/facebookresearch/fairseq/tree/nllb|  0    | 20240124   |   是  | |
|   GPT-SoVITS |  https://github.com/RVC-Boss/GPT-SoVITS|  0    | 20240124   |   否  | |
|   chain-of-knowledge | https://github.com/DAMO-NLP-SG/chain-of-knowledge|  0    | 20240126   |  是 | |
|  fastapi | fastapi框架迅速学习一下 |  0    | 20240207   |  是 | 用于做项目后端，代码完全版本见:  studylist/projects/sql_twice/web位置，将原始的终端展示改为了网页展示|
|  ~~vue~~ | ~~前端框架迅速学习一下~~ |  ~~0~~    | ~~20240207~~   |  否 | ~~用于做项目前端~~|
| ~~gradio~~ | ~~自动页面生成学习一下 https://zhuanlan.zhihu.com/p/627099870~~ | ~~10~~ | ~~20240218~~| 否| ~~用于做功能展示~~|
| knowledge+llm | https://github.com/Zach-PineappleMan/KG-LLM-Papers | 0 | 20240305|  是| 用于探索与创新|
| streamlit | https://zhuanlan.zhihu.com/p/670124993 | 15 | 20240312| 是| 用于做功能展，见studylist/projects/streamlit_project（补：是后续所有项目，都会以该界面进行展示）|
| 扫描版pdf读取的10种方式与感悟 | https://blog.csdn.net/weixin_45934622/article/details/130845137 |20230314 | 用于获取文本数据| 是|见studylist/projects/read_pdf |
| chat4question | - |10| 20230325 | 是 | 用于进行题库扩充见studylist/projects/chat4question|
| Scaling Instruction-Finetuned Language Models | https://arxiv.org/pdf/2210.11416.pdf |0| 20230326 | 是 |选用了base 很难完成ner任务，见studylist/projects/NERtest|

# 番外

20240111 发现苏剑林搭建了:https://papers.cool/ 这个网站，真开心，每天能刷100个abstract的感觉真开心呀！

20240402 四月是炎热的，夏天快点来吧！！！
