import os
import json
import argparse
from collections import OrderedDict

def filter_jsonl_file(input_path, output_path, fields_to_keep):
    """
    读取一个jsonl文件，每行只保留指定的字段，并写入新文件。

    Args:
        input_path (str): 输入jsonl文件的路径。
        output_path (str): 输出jsonl文件的路径。
        fields_to_keep (list): 希望保留的字段名列表。
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                # 跳过空行
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 创建一个只包含指定字段的新字典
                    # 使用OrderedDict可以确保输出的字段顺序与你指定的顺序一致
                    filtered_data = OrderedDict()
                    
                    # 遍历你想要保留的字段列表
                    for field in fields_to_keep:
                        # 如果原始数据中存在这个字段，就将其添加到新字典中
                        if field in data:
                            filtered_data[field] = data[field]
                            
                    # 将筛选后的字典转换回JSON字符串并写入新文件
                    outfile.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    print(f"警告: 文件 '{input_path}' 的第 {line_num} 行不是有效的JSON，已跳过。")
                except Exception as e:
                    print(f"处理文件 '{input_path}' 第 {line_num} 行时发生未知错误: {e}")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_patt}' 未找到。")
    except Exception as e:
        print(f"处理文件 '{input_path}' 时发生错误: {e}")

def process_directory(input_dir, output_dir, fields_to_keep):
    """
    处理指定目录下的所有 .jsonl 文件。

    Args:
        input_dir (str): 输入文件夹路径。
        output_dir (str): 输出文件夹路径。
        fields_to_keep (list): 希望保留的字段名列表。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            print(f"正在处理: {input_file_path} -> {output_file_path}")
            filter_jsonl_file(input_file_path, output_file_path, fields_to_keep)
            
    print("\n所有 .jsonl 文件处理完毕。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="读取文件夹下的所有jsonl文件，删除指定字段之外的所有字段。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input_dir', 
        type=str, 
        required=True, 
        help="包含源 .jsonl 文件的文件夹路径。"
    )
    
    parser.add_argument(
        '-o', '--output_dir', 
        type=str, 
        required=True, 
        help="用于存放处理后文件的文件夹路径。"
    )
    
    parser.add_argument(
        '-f', '--fields', 
        type=str, 
        required=True, 
        help="需要保留的字段名，用逗号分隔 (例如: 'id,name,content')。"
    )
    
    args = parser.parse_args()
    
    # 将逗号分隔的字符串转换为列表
    fields_to_keep = [field.strip() for field in args.fields.split(',')]
    
    process_directory(args.input_dir, args.output_dir, fields_to_keep)