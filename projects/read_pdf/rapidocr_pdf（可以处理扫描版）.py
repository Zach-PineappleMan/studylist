import pandas as pd
from rapidocr_pdf import PDFExtracter

pdf_extracter = PDFExtracter()
pdf_path = 'mybook/data.pdf'

# 假设这个函数调用返回一个列表，每个元素是一页的文本
texts = pdf_extracter(pdf_path)

# 创建一个空的DataFrame
df = pd.DataFrame(columns=['Page', 'Text'])

# 遍历返回的文本列表，将每页的文本添加到DataFrame中
for i, text in enumerate(texts, start=1):
    df = df.append({'Page': f"Page {i}", 'Text': text}, ignore_index=True)

# 保存DataFrame到CSV文件
df.to_csv("extracted_text_per_page.csv", index=False)

print("Texts have been saved to extracted_text_per_page.csv")
