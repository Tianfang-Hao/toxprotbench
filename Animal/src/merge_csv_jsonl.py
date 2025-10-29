# -*- coding: utf-8 -*-

import csv
import json
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from metrics import write_metric

# --- 配置区 ---
# 在这里硬编码您需要操作的字段名
# -------------------------------------------------------------

# 1. 定义对齐主键
CSV_ID_FIELD = 'ID'
JSONL_ID_FIELD = 'id'

# 2. 定义需要从 CSV 文件中提取的列名（字段名）
FIELDS_FROM_CSV = ['ML_Score', 'Prediction']

# 3. 定义需要从 JSONL 文件中提取的字段名
# 来自 JSONL（模型与下游流程的产物）需要带上 assistant（模型输出）
FIELDS_FROM_JSONL = ['id', 'description', 'sequence', 'taxonomy', 'prompt', 'assistant']

# 4. 定义最终输出的 JSONL 文件中的所有字段名及其顺序
#    这个列表的顺序将决定最终JSONL文件中字段的排列顺序。
#    它应该是上面两个列表的并集。
OUTPUT_FIELDS = ['id', 'description', 'sequence', 'taxonomy', 'prompt', 'assistant', 'Prediction', 'ML_Score']

# -------------------------------------------------------------
# --- 配置结束 ---


def merge_files(csv_path, jsonl_path, output_path, metric_file=None):
    """
    从一个CSV文件和一个JSONL文件中提取指定字段，并合并成一个新的JSONL文件。

    Args:
        csv_path (str): 输入的CSV文件路径。
        jsonl_path (str): 输入的JSONL文件路径。
        output_path (str): 输出的JSONL文件路径。
    """
    print(f"开始处理...\n  - 从 CSV '{csv_path}' 提取字段: {FIELDS_FROM_CSV}\n  - 从 JSONL '{jsonl_path}' 提取字段: {FIELDS_FROM_JSONL}")
    
    line_count = 0
    matched = 0
    jsonl_missing_in_csv = 0
    csv_missing_in_jsonl = 0
    try:
        # 预检查：统计 JSONL 行数与 CSV 数据行数，用于顺序对齐健康检查
        jsonl_lines = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f_jsonl_count:
            for _ in f_jsonl_count:
                jsonl_lines += 1
        csv_rows = 0
        with open(csv_path, 'r', encoding='utf-8') as f_csv_count:
            csv_reader_count = csv.DictReader(f_csv_count)
            for _ in csv_reader_count:
                csv_rows += 1

        if jsonl_lines != csv_rows:
            print(f"提示: JSONL 行数({jsonl_lines}) 与 CSV 行数({csv_rows}) 不一致，将按 {JSONL_ID_FIELD}<->{CSV_ID_FIELD} 键进行合并。")

        # 读 CSV 到字典：ID -> row
        with open(csv_path, 'r', encoding='utf-8') as f_csv:
            csv_reader = csv.DictReader(f_csv)
            csv_by_id = {}
            for row in csv_reader:
                cid = row.get(CSV_ID_FIELD)
                if cid is not None:
                    csv_by_id[str(cid)] = row

        used_csv_ids = set()

        # 读取 JSONL，按 JSONL 顺序输出，并用 ID 关联 CSV 字段
        with open(jsonl_path, 'r', encoding='utf-8') as f_jsonl, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            for line_idx, json_line in enumerate(f_jsonl, 1):
                if not json_line.strip():
                    continue
                try:
                    json_data = json.loads(json_line.strip())
                except json.JSONDecodeError as e:
                    print(f"警告: 在 JSONL 文件第 {line_idx} 行解析时出错，已跳过该行。错误信息: {e}")
                    continue

                jid = json_data.get(JSONL_ID_FIELD)
                csv_row = csv_by_id.get(str(jid)) if jid is not None else None
                if csv_row is None:
                    jsonl_missing_in_csv += 1

                merged_record = {}
                for field in OUTPUT_FIELDS:
                    if field in FIELDS_FROM_JSONL:
                        merged_record[field] = json_data.get(field)
                    elif field in FIELDS_FROM_CSV:
                        merged_record[field] = csv_row.get(field) if csv_row else None

                if csv_row is not None:
                    matched += 1
                    used_csv_ids.add(str(jid))

                f_out.write(json.dumps(merged_record, ensure_ascii=False) + '\n')
                line_count += 1

        # CSV 中未被使用的 ID 数
        csv_missing_in_jsonl = max(len(csv_by_id) - len(used_csv_ids), 0)
    
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")
        return

    print(f"\n处理完成！共合并 {line_count} 行记录。")
    print(f"结果已保存到: {output_path}")
    write_metric(metric_file, "merge_csv_jsonl_animal", {
        "input_csv": csv_path,
        "input_jsonl": jsonl_path,
        "output_jsonl": output_path,
        "num_merged": line_count,
        "jsonl_lines": jsonl_lines,
        "csv_rows": csv_rows,
        "count_mismatch": (jsonl_lines != csv_rows),
        "matched": matched,
        "jsonl_missing_in_csv": jsonl_missing_in_csv,
        "csv_missing_in_jsonl": csv_missing_in_jsonl,
    })


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从一个CSV文件和一个JSONL文件中提取指定字段，并按预定顺序合并成一个新的JSONL文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_csv", help="输入的CSV文件路径。")
    parser.add_argument("input_jsonl", help="输入的JSONL文件路径。")
    parser.add_argument("output_jsonl", help="合并后的输出JSONL文件路径。")
    parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")
    
    args = parser.parse_args()
    
    merge_files(args.input_csv, args.input_jsonl, args.output_jsonl, metric_file=args.metric_file)


if __name__ == "__main__":
    main()