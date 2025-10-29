#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from metrics import write_metric

def process_jsonl_files_by_finish_reason(input_dir, output_dir, metric_file=None):
    """
    处理输入目录中的所有 .jsonl 文件。

    对于每个文件，它会根据 'finish_reason' 字段的值进行过滤。
    只有当 'finish_reason' 字段的值为 "stop" 时，该行才会被保留。

    在保留的行中，'finish_reason' 字段会从JSON对象中被删除，
    然后修改后的JSON行会被写入到输出目录中一个同名的新文件。

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

                            # 获取 'finish_reason' 字段的内容
                            finish_reason = data.get('finish_reason')

                            # 核心判断逻辑：
                            # 1. 'finish_reason' 字段的值是否为 "stop"
                            if finish_reason == "stop":
                                # 如果满足条件，删除 'finish_reason' 字段
                                # data.pop() 会安全地删除键，如果键不存在也不会报错
                                data.pop('finish_reason', None)
                                
                                # 将修改后的JSON对象转换回字符串，并写入新文件
                                # ensure_ascii=False 保证中文等字符正常写入
                                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                                lines_kept_count += 1
                            # else:
                            #   不满足条件的行被自动抛弃

                        except json.JSONDecodeError:
                            # 如果某一行不是有效的JSON，则打印警告并跳过
                            print(f"   - 警告: 跳过文件 '{filename}' 中的无效JSON行 (行号: {line_number})。")
                            continue
                        except Exception as e:
                            # 捕获其他可能的错误，例如 data 不是一个字典
                            print(f"   - 警告: 处理行 {line_number} 时发生意外错误: {e}")
                            continue

                total_read += lines_processed_count
                total_kept += lines_kept_count
                file_count += 1
                print(f"处理完成: '{filename}'。")
                print(f"   总共处理 {lines_processed_count} 行，保留了 {lines_kept_count} 行。")

            except Exception as e:
                print(f"处理文件 '{input_file_path}' 时发生严重错误: {e}")

    # 写总体指标
    write_metric(metric_file, "filter_no_stop", {
        "input_files": file_count,
        "num_read": total_read,
        "num_kept": total_kept,
        "num_discarded": max(total_read - total_kept, 0),
        "output_dir": output_dir,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="过滤输入目录中的 .jsonl 文件，仅保留 finish_reason 字段为 'stop' 的行，并从输出中删除该字段。"
    )
    parser.add_argument("input_dir", help="输入 .jsonl 文件所在目录路径")
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        help="输出目录路径（默认：与输入目录同级的 <输入目录名>_filtered）"
    )
    parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在或不是一个有效的目录。")
        raise SystemExit(1)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 默认输出目录逻辑 (与原脚本保持一致)
        abs_in = os.path.abspath(input_dir)
        parent = os.path.dirname(abs_in)
        # 稍微修改了默认名称以反映新的逻辑
        output_dir = os.path.join(parent, f"{os.path.basename(abs_in)}_filtered")

    process_jsonl_files_by_finish_reason(input_dir, output_dir, metric_file=args.metric_file)
    print(f"\n所有 .jsonl 文件处理完毕！输出位于: {output_dir}")