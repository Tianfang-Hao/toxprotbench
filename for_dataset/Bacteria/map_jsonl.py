# -*- coding: utf-8 -*-

import json
import os
import re

# ========================= 配置区 (请在此处修改) =========================

# 1. 输入的源 JSONL 文件路径
INPUT_FILE_PATH = 'generated_prompts_from_jsonl.jsonl'

# 2. 输出的目标 JSONL 文件路径
OUTPUT_FILE_PATH = 'VFDB_bac.jsonl'

# 3. 字段映射规则 (支持嵌套路径)
# 格式为: {'源文件中的字段路径': '目标文件中的字段名'}
# - 使用点 (.) 来访问嵌套对象的属性, 例如: 'user.profile.name'
# - 使用 [索引] 来访问数组元素, 例如: 'tags[0].name'
FIELD_MAPPING = {
    'original_data.description': 'description',
    'original_data.sequence': 'sequence',
    'generated_prompt_for_sequence_model': 'prompt',
}

# ========================= 配置区结束 =========================


def get_nested_value(data_dict, path):
    """
    根据给定的路径字符串从嵌套的字典或列表中安全地获取值。
    例如: get_nested_value(data, 'user.profile.name')
          get_nested_value(data, 'tags[0].name')
    如果路径无效或不存在，则返回 None。
    """
    # 将 'tags[0].name' 这种格式转换为 'tags.0.name' 以便统一处理
    path = path.replace('[', '.').replace(']', '')
    keys = path.split('.')
    
    current_value = data_dict
    for key in keys:
        try:
            if isinstance(current_value, list):
                # 如果当前值是列表，则键必须是数字索引
                current_value = current_value[int(key)]
            elif isinstance(current_value, dict):
                # 如果当前值是字典，则使用键进行访问
                current_value = current_value[key]
            else:
                # 如果既不是列表也不是字典，则无法继续深入，路径无效
                return None
        except (KeyError, IndexError, ValueError, TypeError):
            # 任何访问错误（键不存在、索引越界、索引不是数字等）都意味着路径无效
            return None
    return current_value


def process_jsonl_file():
    """
    根据配置区的设置，读取、处理并写入JSONL文件。
    """
    print("脚本开始执行 (支持嵌套结构)...")
    
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"错误: 输入文件 '{INPUT_FILE_PATH}' 未找到。")
        return

    processed_lines = 0
    error_lines = 0
    
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    source_data = json.loads(line)
                    target_data = {}
                    
                    # 遍历映射规则
                    for source_path, target_key in FIELD_MAPPING.items():
                        # 使用辅助函数获取嵌套的值
                        value = get_nested_value(source_data, source_path)
                        
                        # 只有当值成功获取时 (不为 None)，才将其添加到目标字典中
                        # 如果你想保留值为 null 的字段，可以去掉 "is not None" 的判断
                        if value is not None:
                            target_data[target_key] = value
                    
                    if target_data:
                        outfile.write(json.dumps(target_data, ensure_ascii=False) + '\n')
                        processed_lines += 1

                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行不是有效的JSON格式，已跳过。")
                    error_lines += 1
                    continue

    except Exception as e:
        print(f"处理文件时发生严重错误: {e}")
        return

    print("\n脚本执行完毕！")
    print("========================================")
    print(f"  源文件: {INPUT_FILE_PATH}")
    print(f"  目标文件: {OUTPUT_FILE_PATH}")
    print(f"  成功处理并写入行数: {processed_lines}")
    print(f"  跳过的无效JSON行数: {error_lines}")
    print("========================================")


if __name__ == '__main__':
    process_jsonl_file()