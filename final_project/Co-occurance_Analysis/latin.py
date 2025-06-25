import os
import json
import re
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# ----------------------------
# 1. 拉丁语分词 + 清洗
# ----------------------------
def tokenize_latin_clean(text, stopwords=None):
    tokens = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())  # 只保留长度 ≥5 的单词
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens

# 可选：自定义简单拉丁停用词
DEFAULT_STOPWORDS = {
    'et', 'ut', 'in', 'de', 'non', 'cum', 'ad', 'per', 'ex', 'ab', 'atque',
    'aut', 'sed', 'si', 'ut', 'nam', 'quod', 'qui', 'quae', 'hoc',
    'ille', 'illa', 'est', 'sunt', 'esse', 'me', 'te', 'se', 'nos', 'vos','etiam','autem','tamen', 'quidem', 'quoque', 'nihil', 'quibus', 'omnia', 'neque', 'tantum', 'esset', 'igitur', 'nobis', 'illis', 'omnibus', 'eorum', 'usque', 'cuius','sicut', 'potest', 'omnes', 'inter', 'super', 'magis', 'quasi', 'quantum', 'potius',
    'contra', 'causa', 'semper', 'omnis', 'quorum', 'satis', 'licet', 'primum', 'aliquid'

}

# ----------------------------
# 2. 加载 JSON 数据
# ----------------------------
def load_single_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    keywords_set = set()

    for keyword, items in data.items():
        keywords_set.add(keyword.lower())
        for item in items:
            entries.append({
                "file": item.get("file", ""),
                "keyword": keyword.lower(),
                "path": item.get("path", ""),
                "content": item.get("content", "")
            })

    return entries, list(keywords_set)

# ----------------------------
# 3. 构建共现网络并绘图
# ----------------------------
def build_and_save_cooccurrence_graph(entries, keywords, output_dir, window_size=25, top_k=15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for keyword in keywords:
        G = nx.Graph()
        cooccurrence = defaultdict(int)

        for entry in entries:
            if entry["keyword"] != keyword:
                continue

            tokens = tokenize_latin_clean(entry["content"], stopwords=DEFAULT_STOPWORDS)

            for i, token in enumerate(tokens):
                if token == keyword:
                    start = max(0, i - window_size)
                    end = min(len(tokens), i + window_size + 1)
                    for j in range(start, end):
                        if i != j:
                            cooccurrence[tokens[j]] += 1

        top_words = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:top_k]

        print(top_words)

        if not top_words:
            print(f"[跳过] 关键词 '{keyword}' 没有足够的共现词。")
            continue

        # 添加边
        for word, freq in top_words:
            G.add_edge(keyword, word, weight=freq)

        # 构建图和频次映射
        freqs = [freq for _, freq in top_words]
        max_freq = max(freqs)
        min_freq = min(freqs)
        freq_dict = dict(top_words)

        def normalize_node_size(freq):
            return 800 + (3000 - 800) * (freq - min_freq) / (max_freq - min_freq) if max_freq != min_freq else 1500

        def normalize_edge_width(freq):
            return 0.5 + (5.0 - 0.5) * (freq - min(freqs)) / (max(freqs) - min(freqs)) if max(freqs) != min(freqs) else 2

        G.add_node(keyword)
        node_sizes = []
        node_labels = {}

        for node in G.nodes():
            if node == keyword:
                node_sizes.append(3500)
                node_labels[node] = f"{node} (0)"
            else:
                freq = freq_dict.get(node, 1)
                node_sizes.append(normalize_node_size(freq))
                node_labels[node] = f"{node} ({freq})"

        # 绘图
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42, k=0.6)

        edge_weights_raw = [G[u][v]['weight'] for u, v in G.edges()]
        edge_weights = [normalize_edge_width(w) for w in edge_weights_raw]
        node_colors = ['red' if n == keyword else 'skyblue' for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray')
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='serif')

        plt.title(f"{keyword}(Top {top_k})", fontsize=14)
        plt.axis('off')

        save_path = os.path.join(output_dir, f"{keyword}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"[完成] 已保存图像：{save_path}")

# ----------------------------
# 4. 主执行逻辑
# ----------------------------
if __name__ == "__main__":
    json_file = "../searched_result/search_results_latin.json"
    output_folder = "./result/latin_matplot"

    entries, keywords = load_single_json(json_file)
    print(f"加载关键词数量：{len(keywords)}")
    build_and_save_cooccurrence_graph(entries, keywords, output_folder)
