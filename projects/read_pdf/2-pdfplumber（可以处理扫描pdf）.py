import os
import pdfplumber
from PIL import Image
import pandas as pd
import pytesseract


# 设置Tesseract-OCR路径，根据实际安装情况调整
pytesseract.pytesseract.tesseract_cmd = r'D:\ocr\tesseract.exe' # Windows示例

def ocr_page_to_text(image_path):
    """使用OCR技术从给定的图片路径提取文本。"""
    return pytesseract.image_to_string(Image.open(image_path), lang='chi_sim')


# 确保图片保存的目录存在
img_dir = "./imgs"
os.makedirs(img_dir, exist_ok=True)

# 初始化数据存储列表
data = []

with pdfplumber.open("../mybook/data.pdf") as pdf:
    for page_number, page in enumerate(pdf.pages):
        # 提取页面为图片
        im = page.to_image()
        img_path = f"{img_dir}/page_{page_number + 1}.png"
        im.save(img_path)

        # 使用OCR提取图片中的文本
        text = ocr_page_to_text(img_path)

        # 将文本和图片路径添加到数据列表中
        data.append([text, img_path])

# 创建DataFrame
df = pd.DataFrame(data, columns=['OCR Text', 'Image Path'])

# 输出DataFrame或保存到文件
print(df.head(3))
df.to_csv("output_ocr.csv", index=False)
