# -*- coding: utf-8 -*-

import json
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from metrics import write_metric

# --- 配置区 ---
# 在这里硬编码您需要操作的字段名
# -------------------------------------------------------------

# 1. 定义需要从第一个文件中提取的字段名列表
FIELDS_FROM_FILE1 = ['id', 'description', 'sequence', 'taxonomy', 'prompt', 'assistant']

# 2. 定义需要从第二个文件中提取的字段名列表
FIELDS_FROM_FILE2 = ['prediction', 'probability']

# 3. 定义最终输出文件中的所有字段名及其顺序
#    这个列表的顺序将决定最终jsonl文件中字段的排列顺序。
#    它应该是上面两个列表的并集。
OUTPUT_FIELDS = ['id', 'description', 'sequence', 'taxonomy', 'prompt', 'assistant', 'prediction', 'probability']

# -------------------------------------------------------------
# --- 配置结束 ---


def merge_files(input_path1, input_path2, output_path, metric_file=None):
    """
    从两个jsonl文件中提取指定字段并合并成一个新的jsonl文件。

    Args:
        input_path1 (str): 第一个输入文件的路径。
        input_path2 (str): 第二个输入文件的路径。
        output_path (str): 输出文件的路径。
    """
    print(f"开始处理...\n  - 从 '{input_path1}' 提取字段: {FIELDS_FROM_FILE1}\n  - 从 '{input_path2}' 提取字段: {FIELDS_FROM_FILE2}")
    
    line_count = 0
    matched = 0
    jsonl1_missing_in_jsonl2 = 0
    jsonl2_missing_in_jsonl1 = 0
    try:
        # 预检查：统计两个输入文件的行数
        jsonl1_lines = 0
        with open(input_path1, 'r', encoding='utf-8') as f1c:
            for _ in f1c:
                jsonl1_lines += 1
        jsonl2_lines = 0
        with open(input_path2, 'r', encoding='utf-8') as f2c:
            for _ in f2c:
                jsonl2_lines += 1
        if jsonl1_lines != jsonl2_lines:
            print(f"提示: 两个 JSONL 行数不一致：{jsonl1_lines} vs {jsonl2_lines}，将按 id 键进行合并。")

        # 将文件2加载为以 id 为键的字典
        with open(input_path2, 'r', encoding='utf-8') as f2:
            by_id_2 = {}
            for idx2, line2 in enumerate(f2, 1):
                if not line2.strip():
                    continue
                try:
                    data2 = json.loads(line2.strip())
                except json.JSONDecodeError:
                    continue
                key = data2.get('id')
                if key is not None:
                    by_id_2[str(key)] = data2

        used_ids_2 = set()

        # 遍历文件1（保序），按 id 关联文件2
        with open(input_path1, 'r', encoding='utf-8') as f1, \
             open(output_path, 'w', encoding='utf-8') as fout:

            for idx1, line1 in enumerate(f1, 1):
                if not line1.strip():
                    continue
                try:
                    data1 = json.loads(line1.strip())
                except json.JSONDecodeError as e:
                    print(f"警告: 在文件1第 {idx1} 行解析JSON时出错，已跳过。错误信息: {e}")
                    continue

                key = data1.get('id')
                data2 = by_id_2.get(str(key)) if key is not None else None
                if data2 is None:
                    jsonl1_missing_in_jsonl2 += 1

                merged_record = {}
                for field in OUTPUT_FIELDS:
                    if field in FIELDS_FROM_FILE1:
                        merged_record[field] = data1.get(field)
                    elif field in FIELDS_FROM_FILE2:
                        merged_record[field] = data2.get(field) if data2 else None

                if data2 is not None:
                    matched += 1
                    used_ids_2.add(str(key))

                fout.write(json.dumps(merged_record, ensure_ascii=False) + '\n')
                line_count += 1

        jsonl2_missing_in_jsonl1 = max(len(by_id_2) - len(used_ids_2), 0)
    
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")
        return

    print(f"\n处理完成！共合并 {line_count} 行记录。")
    print(f"结果已保存到: {output_path}")
    write_metric(metric_file, "merge_jsonl_bacteria", {
        "input_jsonl": input_path1,
        "input_jsonl2": input_path2,
        "output_jsonl": output_path,
        "num_merged": line_count,
        "jsonl1_lines": jsonl1_lines,
        "jsonl2_lines": jsonl2_lines,
        "count_mismatch": (jsonl1_lines != jsonl2_lines),
        "matched": matched,
        "jsonl1_missing_in_jsonl2": jsonl1_missing_in_jsonl2,
        "jsonl2_missing_in_jsonl1": jsonl2_missing_in_jsonl1,
    })


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从两个jsonl文件中提取指定字段并按预定顺序合并成一个新的jsonl文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file1", help="第一个输入jsonl文件的路径。")
    parser.add_argument("input_file2", help="第二个输入jsonl文件的路径。")
    parser.add_argument("output_file", help="合并后的输出jsonl文件的路径。")
    parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")
    
    args = parser.parse_args()
    
    merge_files(args.input_file1, args.input_file2, args.output_file, metric_file=args.metric_file)


if __name__ == "__main__":
    main()