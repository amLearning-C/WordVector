# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


# # 加载四个不同参数设置的模型
# models = [
#     Word2Vec.load("word2vec_dim200_win5.model"),
#     Word2Vec.load("word2vec_dim200_win7.model"),
#     Word2Vec.load("word2vec_dim300_win5.model"),
#     Word2Vec.load("word2vec_dim300_win7.model")
# ]
#
# # 使用每个模型查找与词语 '黄蓉' 最相似的词语，并打印结果
# for i, model in enumerate(models, 1):
#     print(f"模型 {i}:")
#     similar_words = model.wv.most_similar("武功")
#     print(similar_words)
#     print()



# 下载需要的NLTK资源


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

def get_paragraph_vector(model, paragraph):
    """生成段落向量"""
    words = preprocess_text(paragraph)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    paragraph_vector = np.mean(word_vectors, axis=0)
    return paragraph_vector


paragraphs = [
    "一转头，只见地下明晃晃的撒著十几枚银针，针身镂刻花纹，打造得极是精致。他俯身一枚枚的拾起，握在左掌，忽见银针旁一条大蜈蚣肚腹翻转，死在地下。他觉得有趣，低头细看，见地下蚂蚁死了不少，数步外尚有许多蚂蚁正在爬行。他拿一枚银针去拨弄几下，那几只蚂蚁兜了几个圈子，便即翻身僵毙，连试几只小虫都是如此。",
    "天灵星一抬头，和古浊飘那锐利的目光撞个正着，他心中一动，升起一个念头，猛的走前两步，一把拍向古浊飘的肩头，笑道：“一掷千金无吝色，神州谁是真豪杰，公子的确是快人。”古浊飘眼神一动，已觉一股极强的力道压了下来，暗忖道：“这老儿倒是个内家高手。”随即微微一笑，在这力道尚未使满之际，伸出手去，像是去拉天灵星的膀子，口中却笑道：“孙老英雄过奖了。”",
    "2014年乌克兰危机开启了俄罗斯与乌克兰双边关系恶化的进程，尤其是克里米亚全民公投并入俄罗斯之后使得两国关系长期处于敌对状态，再加上乌东地区的分离倾向以及俄罗斯对乌东两个“共和国”的支持，加深了俄乌边境军事对抗局面，这些都是今日俄乌冲突爆发的前奏与预演。"
    #"过了岭，黄昏时分已抵三道沟。那是一个不大不小的市镇。骡夫道：“三道沟就只一家安通客栈。”进了镇，镖行和曾图南一行人都投安通客栈。塞外处处荒凉，那客店土墙泥地，也就简陋得很。童兆和不见店里伙计出来迎接，大骂：“店小二都死光了么？我操你十八代祖宗！”李沅芷眉头一皱，她可从来没听人敢当着她面骂这些粗话。"

    ]

    # 加载模型
models = [
        Word2Vec.load("word2vec_dim200_win5.model"),
        Word2Vec.load("word2vec_dim200_win7.model"),
        Word2Vec.load("word2vec_dim300_win5.model"),
        Word2Vec.load("word2vec_dim300_win7.model")
    ]

    # 生成段落向量
paragraph_vectors = {i: [get_paragraph_vector(model, paragraph) for paragraph in paragraphs] for i, model in enumerate(models)}

    # 计算段落向量之间的余弦相似度
similarity_matrices = {i: cosine_similarity(paragraph_vectors[i]) for i in range(len(models))}

    # 输出相似度矩阵
for i, matrix in similarity_matrices.items():
    print(f"Model {i}:")
    print(matrix)

