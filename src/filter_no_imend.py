import os
import json
import argparse
import time
from metrics import write_metric, metric_timer

def process_jsonl_files(input_dir, output_dir, metric_file=None):
    """
    处理输入目录中的所有 .jsonl 文件。

    对于每个文件，它会根据 'assistant' 字段的内容进行过滤。
    只有当 'assistant' 字段是一个以 '<|im_end|>' 结尾的字符串时，
    该行才会被保留。

    过滤后的行会被写入到输出目录中一个同名的新文件。

    参数:
        input_dir (str): 包含输入 .jsonl 文件的目录路径。
        output_dir (str): 用于保存输出文件的目录路径。
    """
    # 1. 确保输出目录存在，如果不存在则创建
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录 '{output_dir}' 已准备就绪。")
    except OSError as e:
        print(f"错误：创建输出目录 '{output_dir}' 失败: {e}")
        return

    total_read = 0
    total_kept = 0
    file_count = 0

    # 2. 遍历输入目录下的所有文件和子目录
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            print(f"\n正在处理文件: {input_file_path}")

            lines_kept_count = 0
            lines_processed_count = 0

            try:
                # 3. 使用 'with' 语句安全地打开输入和输出文件
                with open(input_file_path, 'r', encoding='utf-8') as infile, \
                     open(output_file_path, 'w', encoding='utf-8') as outfile:

                    # 4. 逐行读取和处理
                    for line_number, line in enumerate(infile, 1):
                        lines_processed_count += 1
                        try:
                            # 解析JSON行
                            data = json.loads(line)

                            # 获取 'assistant' 字段的内容
                            assistant_content = data.get('assistant')

                            # 核心判断逻辑：
                            # 1. 'assistant' 字段存在且其值为字符串
                            # 2. 该字符串以 '<|im_end|>' 结尾
                            if isinstance(assistant_content, str) and assistant_content.endswith('<|im_end|>'):
                                # 如果满足条件，将原始行（包含换行符）写入新文件
                                outfile.write(line)
                                lines_kept_count += 1
                            # else:
                            #   不满足条件的行被自动抛弃

                        except (json.JSONDecodeError, KeyError):
                            # 如果某一行不是有效的JSON，或者缺少'assistant'键，则打印警告并跳过
                            print(f"  - 警告: 跳过文件 '{filename}' 中的无效行 (行号: {line_number})。")
                            continue

                print(f"处理完成: '{filename}'。")
                print(f"  总共处理 {lines_processed_count} 行，保留了 {lines_kept_count} 行。")

                # 汇总
                total_read += lines_processed_count
                total_kept += lines_kept_count
                file_count += 1

            except Exception as e:
                print(f"处理文件 '{input_file_path}' 时发生严重错误: {e}")

    # 写出总体指标
    write_metric(metric_file, "filter_no_imend", {
        "input_files": file_count,
        "num_read": total_read,
        "num_kept": total_kept,
        "num_discarded": max(total_read - total_kept, 0),
        "output_dir": output_dir,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="过滤输入目录中的 .jsonl 文件，仅保留 assistant 字段以 '<|im_end|>' 结尾的行。"
    )
    parser.add_argument("input_dir", help="输入 .jsonl 文件所在目录路径")
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        help="输出目录路径（默认：与输入目录同级的 filtered_<输入目录名>）"
    )
    parser.add_argument(
        "--metric_file", type=str, default=None, help="指标输出 JSONL 文件"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在或不是一个有效的目录。")
        raise SystemExit(1)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        abs_in = os.path.abspath(input_dir)
        parent = os.path.dirname(abs_in)
        output_dir = os.path.join(parent, f"{os.path.basename(abs_in)}_filtered")

    #     # 基于 input_folder 生成默认的 output_folder
    # if not args.output_folder:
    #     args.output_folder = f"{args.input_folder.rstrip('/')}_esm"


    process_jsonl_files(input_dir, output_dir, metric_file=args.metric_file)
    print("\n所有 .jsonl 文件处理完毕！")