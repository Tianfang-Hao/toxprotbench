import json
import hashlib
import argparse

def generate_unique_id(text: str) -> str:
    """
    根据给定的文本生成一个SHA-256哈希值作为唯一ID。

    Args:
        text: 用于生成哈希的输入字符串。

    Returns:
        64个字符的十六进制哈希字符串。
    """
    # 确保输入是字符串
    if not isinstance(text, str):
        text = str(text)
    
    # 将字符串编码为UTF-8字节，然后计算哈希值
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_jsonl(input_file: str, output_file: str, prompt_field: str = 'prompt', id_field: str = 'id'):
    """
    处理JSONL文件，为每一行基于prompt字段的值添加一个唯一ID。

    Args:
        input_file: 输入的JSONL文件路径。
        output_file: 输出的JSONL文件路径。
        prompt_field: 用作生成ID依据的字段名（默认为 'prompt'）。
        id_field: 新增的ID字段名（默认为 'id'）。
    """
    print(f"开始处理文件: {input_file}")
    processed_lines = 0
    skipped_lines = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # 尝试解析每一行JSON
                try:
                    data = json.loads(line)
                    
                    # 检查指定的prompt字段是否存在且不为空
                    if prompt_field in data and data[prompt_field]:
                        # 基于prompt字段的值生成ID
                        unique_id = generate_unique_id(data[prompt_field])
                        
                        # 将ID添加到JSON对象中
                        data[id_field] = unique_id
                        
                        # 将更新后的对象写回新文件，确保中文字符不被转义
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        processed_lines += 1
                    else:
                        # 如果prompt字段不存在或为空，可以选择跳过或原样写入
                        # 这里我们选择原样写入并打印警告
                        print(f"警告: 第 {processed_lines + skipped_lines + 1} 行缺少或空的 '{prompt_field}' 字段，已原样写入。")
                        outfile.write(line) # 写入原始行
                        skipped_lines += 1

                except json.JSONDecodeError:
                    print(f"警告: 第 {processed_lines + skipped_lines + 1} 行不是有效的JSON，已跳过。")
                    skipped_lines += 1

        print("\n处理完成！")
        print(f"总计处理行数: {processed_lines}")
        print(f"跳过/原样写入行数: {skipped_lines}")
        print(f"结果已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="为JSONL文件中的每一行基于prompt字段生成唯一ID。")
    parser.add_argument('-i', '--input', required=True, help="输入的JSONL文件路径")
    parser.add_argument('-o', '--output', required=True, help="输出的JSONL文件路径")
    parser.add_argument('--prompt-field', default='prompt', help="作为ID生成依据的字段名 (默认为: 'prompt')")
    parser.add_argument('--id-field', default='id', help="新生成的ID字段名 (默认为: 'id')")
    
    args = parser.parse_args()
    
    # 执行处理函数
    process_jsonl(args.input, args.output, args.prompt_field, args.id_field)