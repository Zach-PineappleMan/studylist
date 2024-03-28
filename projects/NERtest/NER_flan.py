# -*- coding: utf-8 -*-

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("./models")
model = T5ForConditionalGeneration.from_pretrained("./models")

english_ner = """Google LLC is a company focused on internet-related services and products, including online advertising technologies, search engine, cloud computing, software, and hardware. It was founded by Sergey Brin and Larry Page in 1998.
[Company]: Google, [Founded]: 1998, [Founders]: Sergey Brin, Larry Page

Netflix, Inc. is an American subscription streaming service and production company that allows members to watch movies and television shows on the internet. It was founded by Reed Hastings and Marc Randolph in 1997.
[Company]: Netflix, [Founded]: 1997, [Founders]: Reed Hastings, Marc Randolph

Spotify Technology S.A. is a Swedish audio streaming and media services provider, offering digital copyright protected music and podcasts, including access to millions of songs. It was founded by Daniel Ek and Martin Lorentzon in 2006.
[Company]: Spotify, [Founded]: 2006, [Founders]: Daniel Ek, Martin Lorentzon

Intel Corporation is an American multinational corporation and technology company known for designing and manufacturing semiconductor chips. It was founded by Gordon Moore and Robert Noyce in 1968.
[Company]: Intel, [Founded]: 1968, [Founders]: Gordon Moore, Robert Noyce

Apple Inc. is a multinational company that makes personal computers, mobile devices, and software. Apple was started in 1976 by Steve Jobs and Steve Wozniak.
"""


input_ids = tokenizer(english_ner, return_tensors="pt").input_ids
outputs = model.generate(input_ids,max_length=60)
print(outputs)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))



yuenan_ner = """Google LLC là một công ty tập trung vào các dịch vụ và sản phẩm liên quan đến internet, bao gồm công nghệ quảng cáo trực tuyến, công cụ tìm kiếm, điện toán đám mây, phần mềm và phần cứng. Công ty được Sergey Brin và Larry Page thành lập vào năm 1998.
[Công Ty]: Google, [Năm Thành Lập]: 1998, [Người Sáng Lập]: Sergey Brin, Larry Page

Facebook, Inc., nay được biết đến với tên Meta Platforms, Inc., Công ty được Mark Zuckerberg cùng với các bạn cùng phòng đại học và sinh viên Đại học Harvard thành lập vào năm 2004.
[Công Ty]: Facebook, [Năm Thành Lập]: 2004, [Người Sáng Lập]: Mark Zuckerberg, Eduardo Saverin, Andrew McCollum, Dustin Moskovitz, Chris Hughes

Apple Inc. là một công ty đa quốc gia sản xuất máy tính cá nhân, thiết bị di động, và phần mềm. Apple được Steve Jobs và Steve Wozniak khởi xướng vào năm 1976.
"""
input_ids = tokenizer(yuenan_ner, return_tensors="pt").input_ids
outputs = model.generate(input_ids,max_length=60)
print(outputs)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))

