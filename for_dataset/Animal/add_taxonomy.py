import json
import os

# ==================== 配置区 START ====================

# 1. 输入的 JSONL 文件路径
INPUT_FILE = 'animal_ID.jsonl'

# 2. 输出的 JSONL 文件路径 (如果文件已存在，将会被覆盖)
OUTPUT_FILE = 'animal_complete.jsonl'

# 3. 要添加的新字段的名称 (Key)
NEW_FIELD_NAME = 'taxonomy'

# 4. 要添加的新字段的值 (Value)
#    值的类型可以是字符串、数字、布尔值、列表或字典等
#    - 字符串: 'some_value'
#    - 数字: 123
#    - 布尔值: True
#    - 列表: [1, 2, 'apple']
#    - 字典 (嵌套对象): {'source': 'manual_add'}
NEW_FIELD_VALUE = 'Animal'

# ===================== 配置区 END =====================


def add_field_to_jsonl():
    """
    读取 JSONL 文件，为每一行添加一个新字段，并写入到新文件中。
    """
    print("脚本开始执行...")
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 '{INPUT_FILE}' 不存在。")
        return

    processed_lines = 0
    error_lines = 0

    try:
        # 使用 'with' 语句安全地打开文件
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

            # 逐行读取输入文件
            for i, line in enumerate(infile, 1):
                # 去除行尾的换行符和空白
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue

                try:
                    # 将 JSON 字符串解析为 Python 字典
                    data = json.loads(line)
                    
                    # 添加新的字段和值
                    # 如果字段已存在，它的值将会被更新
                    data[NEW_FIELD_NAME] = NEW_FIELD_VALUE
                    
                    # 将更新后的字典转换回 JSON 字符串
                    # ensure_ascii=False 保证中文字符不被转义
                    new_line = json.dumps(data, ensure_ascii=False)
                    
                    # 将新的 JSON 字符串写入输出文件，并添加换行符
                    outfile.write(new_line + '\n')
                    processed_lines += 1

                except json.JSONDecodeError:
                    print(f"警告：第 {i} 行不是有效的 JSON 格式，已跳过。内容: '{line}'")
                    error_lines += 1
    
    except IOError as e:
        print(f"文件处理时发生错误: {e}")
        return

    print("\n脚本执行完成！")
    print(f"总共处理行数: {processed_lines}")
    if error_lines > 0:
        print(f"格式错误并跳过的行数: {error_lines}")
    print(f"结果已保存到: '{OUTPUT_FILE}'")


if __name__ == '__main__':
    add_field_to_jsonl()