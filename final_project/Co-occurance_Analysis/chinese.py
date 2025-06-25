import os
import json
import jieba
import jieba.posseg as pseg
from collections import defaultdict

# === 可配置项 ===
DATA_FILE = "../searched_result/chinese.json"         # <-- 第一段代码的输出文件
OUTPUT_DIR = "./result/chinese"                      # 输出目录
USE_STOPWORDS = True                               # 是否启用停用词过滤
WINDOW_SIZE = 5                                 # 共现窗口大小

# === 加载停用词 ===
stopwords = set()
if USE_STOPWORDS:
    try:
        with open("./stopwords.txt", encoding="utf-8") as f:
            stopwords = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print("未找到 stopwords.txt，将不使用停用词。")

# === 读取处理后的 JSON 数据（如 latin.json）===
def load_cleaned_data(json_path):
    entries = []
    keywords = set()

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

        for kw, item in data.items():
            keywords.add(kw)
            for content in item["contents"]:
                entries.append({
                    "keyword": kw,
                    "content": content
                })

    return entries, list(keywords)


# === 提取共现词 ===
def extract_co_occurrence(entries, keyword):
    co_occurrence = defaultdict(int)

    for entry in entries:
        if keyword != entry.get("keyword"):
            continue

        text = entry.get("content", "")
        words_pos = [(w.word, w.flag) for w in pseg.cut(text)]

        valid_words = [
            w for w, flag in words_pos
            if len(w) >= 1 and w not in stopwords and (flag.startswith('n') or flag.startswith('v'))
        ]

        for i in range(len(valid_words)):
            center_word = valid_words[i]
            if center_word == keyword:
                for j in range(max(0, i - WINDOW_SIZE), min(len(valid_words), i + WINDOW_SIZE + 1)):
                    if j != i:
                        pair = tuple(sorted([keyword, valid_words[j]]))
                        co_occurrence[pair] += 1

    return co_occurrence


# === 保存共现词到 TXT 文件 ===
def save_top_co_occurrence(co_occurrence, keyword, output_dir, top_k=20):
    if not co_occurrence:
        print(f"无共现数据：{keyword}")
        return

    top_items = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:top_k]

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{keyword}_cooccurrence.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"关键词「{keyword}」的Top {top_k} 共现词：\n")
        for (w1, w2), count in top_items:
            f.write(f"{w1} - {w2}: {count} 次\n")

    print(f"已保存：{output_file}")


# === 主函数 ===
if __name__ == "__main__":
    entries, keywords = load_cleaned_data(DATA_FILE)
    print(f"加载处理后语料：{len(entries)} 条，关键词数：{len(keywords)}")

    for kw in keywords:
        print(f"正在分析关键词：{kw}")
        co = extract_co_occurrence(entries, kw)
        save_top_co_occurrence(co, kw, OUTPUT_DIR)
