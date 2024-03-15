from langchain.document_loaders import PyPDFLoader
import csv

# 加载PDF文件
loader = PyPDFLoader("../mybook/data.pdf")
docs = loader.load()

# 将文档内容写入CSV文件
with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for doc in docs:
        writer.writerow([doc.page_content])
