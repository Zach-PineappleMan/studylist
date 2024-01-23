# -*- coding: utf-8 -*-

import pandas as pd
import duckdb
from openai import OpenAI
import time
import os
import mysql.connector

from pprint import pprint
"""
----------------------- 1 -------------------
基础信息：
数据库数据读入和大模型准备工作

"""
# 数据库
DATABASE_TYPE="mysql"
DATABASE_USER=""
DATABASE_PASSWORD=""
DATABASE_HOST="
DATABASE_PORT=
DATABASE_NAME=""

# 表
table = ""

# 大模型key
# version:<= 0.28 openai.api_key = "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"
client=OpenAI(api_key= "")

# 用户输入
query = ""
# query = ""

# 读入数据库信息
conn = mysql.connector.connect(host=DATABASE_HOST,user=DATABASE_USER, password=DATABASE_PASSWORD, database=DATABASE_NAME,port = DATABASE_PORT)
# 查询表的基本信息
query_info = """
SELECT 
    COLUMN_NAME, 
    DATA_TYPE, 
    IS_NULLABLE, 
    COLUMN_DEFAULT, 
    COLUMN_COMMENT
FROM 
    INFORMATION_SCHEMA.COLUMNS 
WHERE 
    TABLE_SCHEMA = 'agency' 
    AND TABLE_NAME = 'face_post_tag_v3';
"""
table_info = pd.read_sql(query_info, conn)

# """
# ----------------------- 2 -------------------
# 构建输入的prompt：
# """

prompt_template = """
根据以下带有其列和注释的 SQL 表格，你的任务是根据用户的请求编写查询语句。\n
CREATE TABLE {} ({}) \n
列详情和备注信息\n{}
编写一个 SQL 查询语句,返回且只返回sql代码块，按照互动量降序排序、禁用DISTINCT，末尾不要分号{}
案例1:小红书有多少个博主提到烟酰胺？ 回答：SELECT COUNT(DISTINCT post_id) FROM face_post_tag_v3 WHERE platform = 2 AND ARRAY_CONTAINS(ingredients，"烟酷胺”)
案例2:抖音哪个博主（post_id）讲提亮讲得比较好？回答：SELECT post_id, engagement_count FROM face_post_tag_v3 WHERE platform = 1  AND ARRAY_CONTAINS(ingredients, '提亮') ORDER BY engagement_count DESC, post_publish_time DESC
"""

# 转换 table_info DataFrame 为字符串格式
def format_table_info(df):
    info_str = ""
    for index, row in df.iterrows():
        info_str += "{}: {}, ".format(row['COLUMN_NAME'], row['DATA_TYPE'])
        if row['COLUMN_COMMENT']:
            info_str += "Comment: '{}' ".format(row['COLUMN_COMMENT'])
        info_str += "\n"
    return info_str

# 不同的表，返回不同的描述词：table_name:表名；table_info：表信息；query：用户问题
def sql_prompt_generator(table_name, table_info, query=''):
    formatted_table_info = format_table_info(table_info)
    col_names = str(list(table_info['COLUMN_NAME'])).replace('[', '').replace(']', '')
    prompt = prompt_template.format(table_name, col_names, formatted_table_info,query)
    return prompt


# 假设 table_info 是之前从数据库中获取的 DataFrame
p = sql_prompt_generator(table_name=table, table_info=table_info)
p = p.replace('unknown','array<varchar(100)>')
# print(p)

# version<=2.8
# def get_chat_response(p,query):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",  # 或者你想使用的其他聊天模型
#             messages=[
#                 {"role": "system", "content": p},
#                 {"role": "user", "content": query}
#             ]
#         )
#         return response.choices[0].message['content']
#     except Exception as e:
#         return str(e)

def get_chat_response(prompt_sys,query_user):
    try:
        response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "system", "content": prompt_sys},
              {"role": "user", "content": query_user},
                 ]
            )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

response = get_chat_response(p,query)

if 'sql' in response:
    sql = response[7:-3]
else:
    sql = response

print("----question----")
print(query)
print("\n------ sql -------")
print(sql)
print("\n----- search ------")
print("前几行数据如下：")
# 执行一下
conn = mysql.connector.connect(host=DATABASE_HOST,
                               user=DATABASE_USER,
                               password=DATABASE_PASSWORD,
                               database=DATABASE_NAME,
                               port = DATABASE_PORT)

result = pd.read_sql(sql, conn)
print(result.head())
conn.close()

print("")
print("----- ans ------")
table_str = result.head().to_string(index=False)
new_prompt= f"""
你将扮演一个人工客服，
以下表格提供的搜索结果就是根据用户的问题在数据库中查询的结果，特别注意podt_id可以代表博主名称，请根据这个进行回答。
{table_str}
请很礼貌地用简洁明了的语句回答问题。
"""
# response = get_chat_response(p,query)
response = get_chat_response(new_prompt,query)
print(response)


