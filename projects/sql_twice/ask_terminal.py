# -*- coding: utf-8 -*-


import pandas as pd
from openai import OpenAI
import time
import os
import mysql.connector
from datetime import datetime
from pprint import pprint
import re
import config

# 数据库
DATABASE_TYPE = config.DATABASE_TYPE
DATABASE_USER = config.DATABASE_USER
DATABASE_PASSWORD = config.DATABASE_PASSWORD
DATABASE_HOST = config.DATABASE_HOST
DATABASE_PORT = config.DATABASE_PORT
DATABASE_NAME = config.DATABASE_NAME

table = config.table

conn = mysql.connector.connect(host=DATABASE_HOST, user=DATABASE_USER, password=DATABASE_PASSWORD,
                               database=DATABASE_NAME, port=DATABASE_PORT)
# 查询表的基本信息
query_info = "SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \" " + DATABASE_NAME + " \" AND TABLE_NAME = \" " + table + "\";"
table_info = pd.read_sql(query_info, conn)
# 修改 COLUMN_COMMENT 将平台增加上具体说明，解释一下互动数的含义以及将unkonwn数据类型改成array
table_info = table_info.replace('平台', "'douyin'指的是抖音平台,'xiaohongshu'指的是小红书平台").replace('互动数',"互动数指的是点赞、评论、分享/转发、收藏数量之和，火不火、好不好、优秀程度等都看这个").replace(
    'unknown', 'array<varchar(100)>')

# 表
table = config.table
# apikey
client = OpenAI(api_key=config.OPENAI_API_KEY)
# prompt_模板1
prompt_template = config.prompt_template
new_prompt = config.new_prompt

def format_table_info(df):
    info_str = ""
    for index, row in df.iterrows():
        info_str += "{}: {}, ".format(row['COLUMN_NAME'], row['DATA_TYPE'])
        if row['COLUMN_COMMENT']:
            info_str += "Comment: '{}' ".format(row['COLUMN_COMMENT'])
        info_str += "\n"
    # print(info_str)
    return info_str

# 将table_info填写到描述词中
# 不同的表，返回不同的描述词：table_name:表名；table_info：表信息；query：目前为空
def sql_prompt_generator(table_name, table_info, query=''):
    formatted_table_info = format_table_info(table_info)
    col_names = str(list(table_info['COLUMN_NAME'])).replace('[', '').replace(']', '')
    prompt = prompt_template.format(table_name, col_names, formatted_table_info, query)
    return prompt

def get_sql_language(q):
    # 构建system部分成功！完整版的输入为p，加入上下文功能后，就不需要了应该
    p = sql_prompt_generator(table_name=table, table_info=table_info)
    # 提问建议加上
    # 获取当前日期和时间
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day

    response = get_chat_response(f"本次提问日期是：{year}年{month}月{day}日 \n" + p, q)

    def get_sql(text):
        sql_match = re.search(r"```sql\n(.*)\n```", text, re.DOTALL)
        return sql_match.group(1) if sql_match else None

    if 'sql' in response:
        sql = get_sql(response)
    else:
        sql = response
    # 使用正则表达式替换 LIMIT 后的任何数字为 5

    # 检查是否存在 'LIMIT \d+' 模式
    if re.search(r"LIMIT \d+", sql):
        # 如果存在，则替换现有的 LIMIT 数字为 6
        sql = re.sub(r"LIMIT \d+", "LIMIT 6", sql)
    else:
        # 移除可能存在的末尾分号
        sql = sql.strip().rstrip(';')
        # 如果不存在，则在 SQL 语句末尾添加 'LIMIT 6'
        sql += " LIMIT 6"

    return sql

def get_ans(table_str,query):
    new_prompt = config.new_prompt
    new_query = f"表格:\n " + table_str + "\n 问题：\n" + query
    response = get_chat_response(new_prompt, new_query)
    return response

def get_chat_response(prompt_sys, prompt_user):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user},
            ]
        )
        # print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(str(e) + "\n 5秒后重试", end='')
        time.sleep(1)
        print(".", end='')
        time.sleep(2)
        print(".", end='')
        time.sleep(2)
        print(".", end='')
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt_sys},
                    {"role": "user", "content": prompt_user},
                ]
            )
            # print(response)
            if "Empty DataFrame" in response.choices[0].message.content:
                print("为空，再试一次")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt_sys},
                        {"role": "user", "content": prompt_user},
                    ]
                )
            return response.choices[0].message.content
        except Exception as e1:
            return str(e1)

def chat_sql_terminal():
    global conn
    # 读入数据库信息

    turns_q1 = []  # turns += [question] + [result]  # 只有这样迭代才能连续提问理解上下文
    text_q1 = ""  # 将历史数据作为文本数据进行提问

    turns_q2 = []  # turns += [question] + [result]  # 只有这样迭代才能连续提问理解上下文
    text_q2 = ""  # 将历史数据作为文本数据进行提问

    # 输入exit 退出
    while True:
        query = input("输入句子：")# "最近一个月急速增长的产品是什么？（就是近五个月而言它的互动量一般，但是最近一个月互动量很高）"
        if query == "exit":
            conn.close()
            return
        else:
            query_processed = query.replace("哪个", "哪三个")

        # 展示一下提问问题
        print("----question----")
        print(query)

        # 进行首次提问(使用处理后的会比较好)
        prompt = text_q1 + "\n\n" + query_processed
        last_result_q1 = get_sql_language(prompt)
        print("\n----sql----")
        print(last_result_q1)

        # 展示一下查询结果
        try:
            result = pd.read_sql(last_result_q1, conn)
            table_str = result.head(6).to_string(index=False)
            print("\n----table----")
            print(table_str)
        except Exception as e_x:
            print("sql出错，再试一次")
            e_x= str(e_x)
            last_result_q1 = get_sql_language(prompt+"请分析为什么会出现这个错误，并重写进行思考，只要输出思考后、修改后的SQL语句代码块："+e_x)
            print("\n----new sql----")
            print(last_result_q1)
            result = pd.read_sql(last_result_q1, conn)
            table_str = result.head(6).to_string(index=False)
            print("\n---- new table----")
            print(table_str)

        # 记录结果（记录的也是处理后的）
        turns_q1 += [query_processed] + [last_result_q1]
        if len(turns_q1) <= 6:  # 为了防止超过字数限制程序会爆掉，所以提交的话轮语境为5次。
            text_q1 = " ".join(turns_q1)
        else:
            text_q1 = " ".join(turns_q1[-6:])

        # 进行接着提问
        prompt = text_q2 + "\n\n" + query
        # print(prompt)
        last_result_q2 = get_ans(table_str,prompt)

        print("\n----ans----")
        print(last_result_q2)

        turns_q2 += [query_processed] + [last_result_q2]
        if len(turns_q2) <= 6:  # 为了防止超过字数限制程序会爆掉，所以提交的话轮语境为5次。
            text_q2 = " ".join(turns_q2)
        else:
            text_q2 = " ".join(turns_q2[-6:])

    conn.close()
    return

chat_sql_terminal()