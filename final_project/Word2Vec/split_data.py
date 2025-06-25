import json
import random
import os

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 统计关键词并整理 content，若条数 > 10000，随机抽取 10000 条
def extract_and_count(data):
    result = {}
    
    # 遍历 JSON 数据，处理每个关键词
    for keyword, entries in data.items():
        contents = [entry['content'] for entry in entries]

        print(len(contents))
        
        # 如果条目数大于1000，则随机抽取1000条
        if len(contents) > 10000:
            new_contents = random.sample(contents, 10000)
        else:
            new_contents = contents

        print(len(new_contents))
        
        # 存储内容和数量
        result[keyword] = {
            'count': len(new_contents),
            'contents': new_contents
        }
    
    return result

# 保存整理后的数据到指定位置
def save_to_file(result, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)



# 示例使用
if __name__ == "__main__":
    # 假设 JSON 数据保存在 "data.json" 文件中
    file_path = '../searched_result/search_results_chinese.json'
    
    # 读取 JSON 数据
    data = read_json(file_path)
    
    # 提取并统计每个关键词的内容
    result = extract_and_count(data)
    
    # 输出结果保存到指定文件
    output_file = '../searched_result/chinese.json'
    save_to_file(result, output_file)
    
    print(f"处理完成，结果已保存到 {output_file}")
