import PyPDF2
import pandas as pd

# 打开PDF文件
with open('../mybook/data.pdf', 'rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)

    # 初始化一个列表来存储每一页的文本
    pages_text = []

    # 逐页读取PDF中的文本
    for page in pdf_reader.pages:
        page_text = page.extract_text() if page.extract_text() else "No text found"
        pages_text.append(page_text)

# 将每页的文本存储到DataFrame中，每行代表一页
df = pd.DataFrame(pages_text, columns=['Text'])

# 保存DataFrame到CSV文件
df.to_csv("extracted_text_per_page.csv", index=False)

print("Text has been saved to extracted_text_per_page.csv")
