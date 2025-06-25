import json
import os
import jieba
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# ==== 配置项 ====
JSON_PATH = "../searched_result/chinese.json"
MODEL_PATH = "./result/chinese_w2v/word2vec.model"
SIMILARITY_OUT = "./result/chinese_w2v/keyword_similarity.txt"
NEIGHBOR_OUT = "./result/chinese_w2v/keyword_neighbors.txt"
SIMILARITY_HEATMAP = "./result/chinese_w2v/keyword_similarity_heatmap.png"
VECTOR_PLOT = "./result/chinese_w2v/word_vectors_pca.png"
TOP_N = 10  # 最近邻个数

# keyword_translation = {
#     "孝": "Filial Piety",
#     "弟": "Brotherly Love",
#     "恭順": "Respect and Obedience",
#     "親": "Parent",
#     "敬": "Respect",
#     "長": "Elder",
#     "父": "Father",
#     "母": "Mother",
#     "親恩": "Parental Love",
#     "孝悌": "Filial Piety and Brotherly Love",
#     "從兄": "Older Brother"
# }

keyword_translation = {
    # 同辈亲属
    "兄": "Older Brother",
    "弟": "Younger Brother",
    "姊": "Older Sister",
    "妹": "Younger Sister",
    "从父": "Paternal Cousin (Father's Side)",  # 堂亲（父之兄弟子女）
    "中表": "Cousins (Maternal and Paternal)",  # 泛指表亲
    # 子女辈
    "子": "Son",
    "女": "Daughter",
    "甥": "Nephew (Sister's Son)",  # 姐妹之子
    "侄": "Nephew (Brother's Son)",  # 兄弟之子
    # 父母辈
    "父": "Father",
    "母": "Mother",
    "伯父": "Paternal Elder Uncle",  # 父之兄
    "姑": "Paternal Aunt",          # 父之姐妹
    "舅": "Maternal Uncle",         # 母之兄弟
    "姨": "Maternal Aunt",          # 母之姐妹
    # 祖辈
    "祖父": "Paternal Grandfather",
    "祖母": "Paternal Grandmother",
    "外祖父": "Maternal Grandfather",
    "外祖母": "Maternal Grandmother",

    "孝": "Filial Piety",
    # "弟": "Brotherly Love",
    "恭順": "Respect and Obedience",
    "親": "Parent",
    "敬": "Respect",
    "長": "Elder",
    # "父": "Father",
    # "母": "Mother",
    "親恩": "Parental Love",
    "孝悌": "Filial Piety and Brotherly Love",
    "從兄": "Older Brother"
}

# ==== 第一步：加载数据 ====
def load_json_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    keywords = list(data.keys())
    sentences = []
    for kw, items in data.items():
        for text in items["contents"]:
            seg = list(jieba.cut(text))
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
    
    english_keywords = [keyword_translation.get(kw, kw) for kw in keywords]


    plt.figure(figsize=(max(10, size * 0.5), max(8, size * 0.5)))
    sns.heatmap(matrix, xticklabels=english_keywords, yticklabels=english_keywords, cmap='YlGnBu', annot=True, fmt=".2f")
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

    # 创建英文标签列表
    english_words = [keyword_translation.get(kw, kw) for kw in words]

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0]+0.01, reduced[i, 1]+0.01, english_words[i], fontsize=12)
    plt.title("The Distribution of Keyword Word Vectors (PCA Dimensionality Reduction)")
    plt.tight_layout()
    
    if out_file:
        plt.savefig(out_file, dpi=300)
    plt.show()

# ==== 主程序 ====
if __name__ == "__main__":
    keywords, sentences = load_json_data(JSON_PATH)
    # keywords = [
    #     "兄", "弟", "姊", "妹", "从父", "中表",
    #     "子","女","甥","侄" ,
    #     "父","母", "伯父","姑","舅","姨",
    #     "祖父", "祖母", "外祖父", "外祖母",
    # ]
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
