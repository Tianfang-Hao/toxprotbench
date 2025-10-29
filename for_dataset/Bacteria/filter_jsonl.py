import json
import argparse

def filter_jsonl_by_field(input_file, output_file, field_name):
    """
    根据 JSONL 文件中指定字段的布尔值筛选行。

    :param input_file: 输入的 .jsonl 文件路径
    :param output_file: 输出的 .jsonl 文件路径
    :param field_name: 用于筛选的字段名 (该字段的值必须是布尔 true)
    """
    lines_read = 0
    lines_written = 0

    print(f"开始处理文件: {input_file}")
    print(f"筛选条件: 字段 '{field_name}' 的值为 true")

    try:
        # 使用 'with' 语句确保文件能被正确关闭
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                lines_read += 1
                try:
                    # 将每一行解析为 Python 字典
                    data = json.loads(line)
                    
                    # 检查字段是否存在且其值是否严格为 True
                    # 使用 .get() 方法可以避免因字段不存在而报错
                    if data.get(field_name) is True:
                        # 如果条件满足，将原始行写入新文件
                        outfile.write(line)
                        lines_written += 1
                
                except json.JSONDecodeError:
                    print(f"警告: 第 {lines_read} 行不是有效的 JSON，已跳过: {line.strip()}")
                except Exception as e:
                    print(f"警告: 处理第 {lines_read} 行时发生错误: {e}")

        print("\n处理完成！")
        print(f"总共读取行数: {lines_read}")
        print(f"成功写入行数: {lines_written}")
        print(f"新的文件已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="筛选 JSONL 文件，仅保留特定字段值为 true 的行。"
    )
    
    # 添加必要的命令行参数
    parser.add_argument("input_file", help="输入的 .jsonl 文件路径")
    parser.add_argument("output_file", help="输出的新 .jsonl 文件路径")
    parser.add_argument("field_name", help="作为筛选依据的字段名")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数
    filter_jsonl_by_field(args.input_file, args.output_file, args.field_name)