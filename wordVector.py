# -*- coding: utf-8 -*-
import jieba
import os
import matplotlib.pyplot as plt
import re
import gensim
from gensim.models import Word2Vec
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文字体，这里以宋体为例

# -*- coding: utf-8 -*-
import jieba
import os
import re
import gensim
from gensim.models import Word2Vec


# 预处理函数：删除隐藏符号、非中文字符和标点符号
def preprocess_text(text):
    # 删除所有的隐藏符号
    cleaned_text = ''.join(char for char in text if char.isprintable())
    # 删除所有的非中文字符
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    # # 删除停用词
    # with open('cn_stopwords.txt', "r", encoding='utf-8') as f:
    #     stopwords = set([line.strip() for line in f])
    #     cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    # # 删除所有的标点符号
    # punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    # cleaned_text = punctuation_pattern.sub(' ', cleaned_text)
    return cleaned_text


folder_path = r"D:/Users/Vincent W/Desktop/研究生/学习/研一下NIP/作业2/jyxstxtqj_downcc.com1/"
output_folder_path = r"D:/Users/Vincent W/Desktop/研究生/学习/研一下NIP/作业2/processed_files/"

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

all_sentences = []  # 存储所有文档的分词结果

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    output_file_path = os.path.join(output_folder_path, f"processed_{file_name}")
    print("处理文件", file_path)

    # 读取文本内容并进行预处理，并进行结巴分词
    with open(file_path, "r", encoding='ansi') as file:
        text = file.read()
        preprocessed_text = preprocess_text(text)
        words = jieba.lcut(preprocessed_text)

    # 将分词结果保存到新的TXT文件中，每个分词用空格隔开
    with open(output_file_path, "w", encoding='utf-8') as output_file:
        output_file.write(' '.join(words))

    all_sentences.append(words)
    print("处理完毕，结果保存到", output_file_path)

# 使用分词结果训练Word2Vec模型
# 定义模型训练函数
def train_word2vec_model(sentences, vector_size, window_size, model_name):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window_size, min_count=1, workers=4, epochs=200)
    model.save(model_name)

# 定义参数组合
dimensions = [200, 300]
windows = [5, 7]

# 遍历参数组合，训练并保存模型
for dim in dimensions:
    for win in windows:
        model_name = f"word2vec_dim{dim}_win{win}.model"
        print(f"训练Word2Vec模型，词向量维度：{dim}，窗口大小：{win}")
        train_word2vec_model(all_sentences, dim, win, model_name)

print("所有模型训练完毕")