import json
import os
import re  # 引入正则表达式库
from gensim.models import Word2Vec
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# ==== 配置项 ====
JSON_PATH = "../searched_result/latin.json"
MODEL_PATH = "./result_1/latin_w2v/word2vec.model"
SIMILARITY_OUT = "./result_1/latin_w2v/keyword_similarity.txt"
NEIGHBOR_OUT = "./result_1/latin_w2v/keyword_neighbors.txt"
SIMILARITY_HEATMAP = "./result_1/latin_w2v/keyword_similarity_heatmap.png"
VECTOR_PLOT = "./result_1/latin_w2v/word_vectors_pca.png"
TOP_N = 10  # 最近邻个数

# ==== 第一步：加载数据 ====
def load_json_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    keywords = list(data.keys())
    sentences = []
    
    # 使用正则表达式进行分词
    word_pattern = re.compile(r'\b[a-zA-Z]+\b')  # 匹配由字母组成的单词
    
    for kw, items in data.items():
        for text in items["contents"]:
            # 使用正则分词，将文本中的单词提取出来
            seg = word_pattern.findall(text)
            sentences.append(seg)
    return keywords, sentences

# ==== 第二步：训练 Word2Vec ====
def train_word2vec(sentences, save_path):
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=1)
    model.save(save_path)
    return model

# ==== 第三步：计算关键词两两相似度 ====
def compute_keyword_similarities(keywords, model):
    results = []
    for w1, w2 in combinations(keywords, 2):
        if w1 in model.wv and w2 in model.wv:
            sim = model.wv.similarity(w1, w2)
            results.append((w1, w2, sim))
    return sorted(results, key=lambda x: x[2], reverse=True)

# ==== 第四步：获取每个关键词的最近邻词汇 ====
def get_nearest_neighbors(keywords, model, top_n=10):
    neighbors = {}
    for kw in keywords:
        if kw in model.wv:
            similar_words = model.wv.most_similar(kw, topn=top_n)
            neighbors[kw] = similar_words
    return neighbors

# ==== 第五步：保存结果 ====
def save_similarity_results(similarities, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for w1, w2, sim in similarities:
            f.write(f"{w1} - {w2}: {sim:.4f}\n")

def save_neighbors(neighbors, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for kw, words in neighbors.items():
            f.write(f"{kw} 的 Top 相似词：\n")
            for word, score in words:
                f.write(f"  {word}: {score:.4f}\n")
            f.write("\n")

# ==== 第六步：可视化相似度热力图 ====
def plot_similarity_heatmap(keywords, model, out_file=None):
    size = len(keywords)
    matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if keywords[i] in model.wv and keywords[j] in model.wv:
                matrix[i][j] = model.wv.similarity(keywords[i], keywords[j])
            else:
                matrix[i][j] = 0.0

    plt.figure(figsize=(max(10, size * 0.5), max(8, size * 0.5)))
    sns.heatmap(matrix, xticklabels=keywords, yticklabels=keywords, cmap='YlGnBu', annot=True, fmt=".2f")
    plt.xticks(rotation=45, ha='right')
    plt.title("Keyword Similarity Heatmap")
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
    plt.show()

# ==== 第七步：可视化词向量分布（PCA降维） ====
def plot_word_vectors(keywords, model, out_file=None):
    words = [kw for kw in keywords if kw in model.wv]
    vecs = [model.wv[w] for w in words]
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vecs)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0]+0.01, reduced[i, 1]+0.01, word, fontsize=12)
    plt.title("The Distribution of Keyword Word Vectors (PCA Dimensionality Reduction)")
    plt.tight_layout()
    
    if out_file:
        plt.savefig(out_file, dpi=300)
    plt.show()

# ==== 主程序 ====
if __name__ == "__main__":
    keywords, sentences = load_json_data(JSON_PATH)

    keywords = [
    # 第0代
    "frater", "soror", "patruelis", "sobrinus", "consobrinus",
    # 第1代
    "filius", "filia",
    # 第2代
    "nepos", "neptis",
    # 第-1代
    "pater", "mater", "patruus", "amita", "avunculus", "matertera",
    # 第-2代
    "avus", "avia",
]

    print(f"关键词数：{len(keywords)}，语料数：{len(sentences)}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model = train_word2vec(sentences, MODEL_PATH)

    similarities = compute_keyword_similarities(keywords, model)
    neighbors = get_nearest_neighbors(keywords, model, top_n=TOP_N)

    save_similarity_results(similarities, SIMILARITY_OUT)
    save_neighbors(neighbors, NEIGHBOR_OUT)

    plot_similarity_heatmap(keywords, model, SIMILARITY_HEATMAP)
    plot_word_vectors(keywords, model, VECTOR_PLOT)

    print(f"关键词相似度保存至：{SIMILARITY_OUT}")
    print(f"关键词最近邻保存至：{NEIGHBOR_OUT}")
    print(f"相似度热力图保存至：{SIMILARITY_HEATMAP}")
    print(f"词向量图保存至：{VECTOR_PLOT}")
