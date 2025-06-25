from bpe import Tokenizer
import os
import json

def read_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    all_json_strings = []

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            json_string = json.dumps(json_data)
            all_json_strings.append(json_string)

    return all_json_strings

def extract_text_from_json(json_data):
    """
    从 JSON 数据中提取文本内容。
    根据你的 JSON 文件结构，可能需要调整这个函数。
    """
    text_content = []
    
    # 如果是字典类型，递归提取值
    if isinstance(json_data, dict):
        for value in json_data.values():
            text_content.extend(extract_text_from_json(value))
    
    # 如果是列表类型，递归提取每个元素
    elif isinstance(json_data, list):
        for item in json_data:
            text_content.extend(extract_text_from_json(item))
    
    # 如果是字符串类型，直接添加到文本内容中
    elif isinstance(json_data, str):
        text_content.append(json_data)
    
    return text_content

def get_json_text_from_folder(folder_path):
    """
    从指定文件夹中的所有 JSON 文件中提取文本内容，并返回一个字符串列表。
    """
    all_json_texts = []

    # 遍历文件夹中的所有 JSON 文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                
                # 读取 JSON 文件内容
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        json_data = json.load(file)
                        
                        # 提取文本内容
                        text_content = extract_text_from_json(json_data)
                        
                        # 将文本内容转换为字符串并添加到列表中
                        all_json_texts.extend(text_content)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON file {file_path}: {e}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

    return all_json_texts

# # 提取base中所有文本
# folder_path = './base'
# json_strings = read_json_files(folder_path)
# json_texts = get_json_text_from_folder(folder_path)
# # 文本被转化为unicode格式
# print(f"number of json strings: {len(json_strings)}")
# # print(json_texts)
# with open('./base.txt', 'w', encoding='utf-8') as file:
#     file.write('\n'.join(json_texts))

# 训练tokenizer
tokenizer = Tokenizer()
with open('./base.txt', 'r', encoding='utf-8') as file:
    text = file.read()
tokenizer.train(text, 1024)

# tokenize the original text
# 可以看到文本被根据词汇表，转化为了数字（编号）
# tokenizer.load("./manual.model")
with open('./train.txt', 'w', encoding='utf-8') as file:
    file.write(f"encoded training text : {tokenizer.encode(text)}\n")
    file.write(f"encode and decode match: {text == tokenizer.decode(tokenizer.encode(text))}")
    
tokenizer.save("base_tokenizer")