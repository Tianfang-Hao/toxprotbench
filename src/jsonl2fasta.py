import json
import argparse
import sys
import os # 导入 os 模块用于处理文件路径
from metrics import write_metric

def convert_jsonl_to_fasta(input_path, output_path, metric_file=None):
    """
    将 JSONL 文件转换为 FASTA 文件。

    从 JSONL 文件的每一行中读取一个 JSON 对象，
    使用 'id' 字段作为 FASTA 描述行，
    使用 'assistant' 字段作为 FASTA 序列。

    Args:
        input_path (str): 输入的 JSONL 文件路径。
        output_path (str): 输出的 FASTA 文件路径。
    """
    print(f"开始转换文件: {input_path} -> {output_path}")
    
    entry_count = 0
    skipped_lines = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    record_id = data.get('id')
                    sequence = data.get('assistant')

                    if isinstance(record_id, (str, int)) and isinstance(sequence, str):
                        header = str(record_id).replace('\n', ' ').replace('\r', '')
                        outfile.write(f">{header}\n")
                        outfile.write(f"{sequence}\n")
                        entry_count += 1
                    else:
                        print(f"警告: 第 {line_num} 行缺少 'id' 或 'assistant' 字段，或者字段类型不正确。已跳过。", file=sys.stderr)
                        skipped_lines += 1

                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行不是有效的 JSON 格式。已跳过。", file=sys.stderr)
                    skipped_lines += 1

        print("\n转换完成！")
        print(f"成功写入 {entry_count} 个 FASTA 条目。")
        if skipped_lines > 0:
            print(f"共跳过 {skipped_lines} 个格式错误的行。")

        # 结构化指标
        write_metric(metric_file, "jsonl2fasta", {
            "input_jsonl": input_path,
            "output_fasta": output_path,
            "num_written": entry_count,
            "num_skipped": skipped_lines,
        })

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_path}' 未找到。", file=sys.stderr)
    except Exception as e:
        print(f"发生未知错误: {e}", file=sys.stderr)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="一个将 JSONL 文件转换为 FASTA 格式文件的脚本。",
        epilog="示例用法: \n"
               "  python jsonl_to_fasta.py input.jsonl (自动生成 output.fasta)\n"
               "  python jsonl_to_fasta.py input.jsonl custom_name.fasta (指定输出文件名)"
    )
    
    # 定义输入文件参数
    parser.add_argument("input_file", help="输入的 JSONL 文件路径")
    # 定义输出文件参数为可选参数
    parser.add_argument("output_file", nargs='?', default=None, help="输出的 FASTA 文件路径 (可选，若不提供则自动生成)")
    parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # ---- 新增逻辑：决定输出文件名 ----
    input_path = args.input_file
    output_path = args.output_file

    if output_path is None:
        # 如果没有提供输出文件名，则根据输入文件名自动生成
        # os.path.splitext 会将 'path/to/file.jsonl' 分割成 ('path/to/file', '.jsonl')
        base_name, _ = os.path.splitext(input_path)
        output_path = base_name + ".fasta"

    # 执行转换函数
    convert_jsonl_to_fasta(input_path, output_path, metric_file=args.metric_file)