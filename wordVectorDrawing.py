# -*- coding: utf-8 -*-
import os
import re
import jieba
import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 确保matplotlib使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def preprocess_text(text):
    cleaned_text = ''.join(char for char in text if char.isprintable())
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', '', cleaned_text)
    with open('cn_stopwords.txt', "r", encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
        cleaned_text = ''.join(char for char in cleaned_text if char not in stopwords)
    punctuation_pattern = re.compile(r'[\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+')
    cleaned_text = punctuation_pattern.sub('', cleaned_text)
    return cleaned_text


def tsne_plot(model, words_num):
    labels = []
    tokens = []

    for word in model.wv.index_to_key[:words_num]:
        tokens.append(model.wv[word])
        labels.append(word)

    tokens = np.array(tokens)  # 转换为NumPy数组

    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = new_values[:, 0]
    y = new_values[:, 1]

    plt.figure(figsize=(14, 10))
    for i in range(words_num):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title(model_description)
    plt.show()


# 加载四个不同参数设置的模型
# 加载四个不同参数设置的模型
models = [
    (gensim.models.Word2Vec.load("word2vec_dim200_win5.model"), "词向量维度=200, 窗口大小=5"),
    (gensim.models.Word2Vec.load("word2vec_dim200_win7.model"), "词向量维度=200, 窗口大小=7"),
    (gensim.models.Word2Vec.load("word2vec_dim300_win5.model"), "词向量维度=300, 窗口大小=5"),
    (gensim.models.Word2Vec.load("word2vec_dim300_win7.model"), "词向量维度=300, 窗口大小=7")
]

# 可视化每个模型
for model, model_description in models:
    print(f"可视化模型：{model_description}")
    tsne_plot(model, 100)
