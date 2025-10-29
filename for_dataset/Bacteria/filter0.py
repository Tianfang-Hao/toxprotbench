#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import os

def filter_fasta(jsonl_path, fasta_path, output_path):
    """
    根据 JSONL 文件中的预测值过滤 FASTA 文件。

    Args:
        jsonl_path (str): 包含预测结果的 JSONL 文件路径。
        fasta_path (str): 原始的 FASTA 文件路径。
        output_path (str): 过滤后输出的新 FASTA 文件路径。
    """
    print(f"[*] 正在从 {jsonl_path} 读取需要保留的蛋白质 ID...")

    ids_to_keep = set()
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 只保留 prediction 值为 1 (或非 0) 的条目
                    if data.get('prediction') == 1:
                        ids_to_keep.add(data['id'])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  [!] 警告: 跳过格式错误的行: {line.strip()} - 错误: {e}")

    except FileNotFoundError:
        print(f"[X] 错误: JSONL 文件未找到: {jsonl_path}")
        return
        
    if not ids_to_keep:
        print("[X] 警告: 没有在 JSONL 文件中找到任何预测值为 1 的条目。将生成一个空的输出文件。")
    else:
        print(f"[+] 完成。共找到 {len(ids_to_keep)} 个需要保留的条目。")

    print(f"[*] 正在处理原始 FASTA 文件: {fasta_path}...")
    
    total_entries = 0
    kept_entries = 0
    
    try:
        with open(fasta_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            current_entry_id = None
            is_keeping_current_entry = False
            
            for line in infile:
                if line.startswith('>'):
                    total_entries += 1
                    # 提取 FASTA 标题行中的 ID
                    # 通常 ID 是 > 后面到第一个空格之前的部分
                    header = line.strip().lstrip('>')
                    current_entry_id = header.split()[0]
                    
                    if current_entry_id in ids_to_keep:
                        is_keeping_current_entry = True
                        kept_entries += 1
                        outfile.write(line)
                    else:
                        is_keeping_current_entry = False
                
                # 如果当前条目需要保留，则写入其序列行
                elif is_keeping_current_entry:
                    outfile.write(line)
                    
    except FileNotFoundError:
        print(f"[X] 错误: FASTA 文件未找到: {fasta_path}")
        return

    print("\n" + "="*30)
    print("      处理完成      ")
    print("="*30)
    print(f"总共处理了 {total_entries} 个 FASTA 条目。")
    print(f"保留了 {kept_entries} 个条目 (预测值为 1)。")
    print(f"删除了 {total_entries - kept_entries} 个条目 (预测值为 0)。")
    print(f"[✓] 已将结果保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="根据 JSONL 文件中的预测值过滤 FASTA 文件，只保留预测值为 1 的条目。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="包含预测结果的 JSONL 文件路径。\n每行应为JSON对象, 例如: {'id': 'prot1', 'prediction': 1, ...}"
    )
    parser.add_argument(
        "fasta_file",
        type=str,
        help="原始的 FASTA 文件路径。"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="输出的过滤后 FASTA 文件路径。\n(默认: 在原始文件名后添加 '_filtered.fasta')"
    )

    args = parser.parse_args()

    # 如果未指定输出路径，则自动生成一个
    if not args.output:
        base_name = os.path.splitext(args.fasta_file)[0]
        args.output = f"{base_name}_filtered.fasta"

    filter_fasta(args.jsonl_file, args.fasta_file, args.output)


if __name__ == "__main__":
    main()