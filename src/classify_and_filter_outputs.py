#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 raw JSONL 中读取 analyze_refuse 已经写入的 refuse/helpful 标签：
1) 保持 raw 原样（不修改 refuse/helpful 判定），可选添加 case/seq_len 等派生信息但不改变原文本和标签；
2) 生成 raw_filtered：仅保留 情况1（refuse=False, helpful=True）的条目，assistant 替换为提取出的蛋白质序列+<|im_end|>。

四种情况定义由 analyze_refuse 完成（refuse/helpful）；本脚本不再变更它们，仅依据标签决定是否写入 filtered。

指标：写入 step=classify_and_filter，统计输入行数、符合情况1并成功提取的条数、失败提取数等；带上 domain=Animal|Bacteria 便于汇总。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from metrics import write_metric


AA_SET = set("ACDEFGHIKLMNPQRSTVWYX")
IM_END_TOKEN = "<|im_end|>"


def extract_longest_aa(text: str, min_len: int) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    # 保持大小写，不做 upper 转换，仅匹配大写 AA
    pattern = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{" + str(min_len) + r",}")
    matches = pattern.findall(text)
    if not matches:
        return None
    return max(matches, key=len)


def is_periodic_loop(seq: str, max_period: int = 10, similarity_threshold: float = 0.98) -> bool:
    """极简周期性检测：若对某个周期 p (1..max_period)，seq 与其平移 p 的相似度 >= 阈值，则认为是循环。"""
    n = len(seq)
    if n == 0:
        return False
    for p in range(1, min(max_period, n) + 1):
        same = 0
        total = n - p
        if total <= 0:
            continue
        # 计算位置 i 与 i-p 是否相同
        for i in range(p, n):
            if seq[i] == seq[i - p]:
                same += 1
        if total > 0 and (same / total) >= similarity_threshold:
            return True
    return False


def is_reasonable_protein(seq: str, min_len: int, max_len: int) -> bool:
    if not seq or len(seq) < min_len or len(seq) > max_len:
        return False
    if any((ch not in AA_SET) for ch in seq):
        return False
    if is_periodic_loop(seq):
        return False
    return True


@dataclass
class Counters:
    total: int = 0
    num_case1: int = 0
    num_written: int = 0
    num_extract_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'num_case1': self.num_case1,
            'num_written': self.num_written,
            'num_extract_failed': self.num_extract_failed,
        }


def process_one_file(input_path: Path, output_filtered_path: Path, metric_file: Optional[str], domain: Optional[str], min_len: int, max_len: int) -> Counters:
    counters = Counters()
    raw_items: List[Dict[str, Any]] = []
    filtered_items: List[Dict[str, Any]] = []

    # 读取 raw
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except Exception:
                # 非法行也计入 total，但只能跳过
                counters.total += 1
                continue

    for obj in raw_items:
        counters.total += 1
        # 使用 analyze_refuse 的标签，不变更判定
        refuse = bool(obj.get('refuse', False))
        helpful = bool(obj.get('helpful', False))
        case = 1 if (not refuse and helpful) else (2 if (not refuse and not helpful) else (3 if (refuse and helpful) else 4))

        assistant_text = obj.get('assistant', '')
        extracted = extract_longest_aa(assistant_text, min_len=min_len)
        seq_valid = is_reasonable_protein(extracted, min_len, max_len) if extracted else False

        # 仅对情况1进行写入 filtered；若无法提取有效序列，计入提取失败
        if case == 1:
            counters.num_case1 += 1
            if seq_valid:
                new_obj = dict(obj)
                new_obj['assistant'] = extracted + IM_END_TOKEN
                new_obj['extracted'] = True
                filtered_items.append(new_obj)
                counters.num_written += 1
            else:
                counters.num_extract_failed += 1

        # raw 保持原文与标签，可选添加提示性派生字段（不改变标签）
        obj['case'] = case
        if extracted:
            obj['seq_len'] = len(extracted)

    # 写回 raw（覆盖原文件，保持所有行）
    with open(input_path, 'w', encoding='utf-8') as f:
        for obj in raw_items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # 写出 filtered（仅 case1 且成功提取）
    output_filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filtered_path, 'w', encoding='utf-8') as f:
        for obj in filtered_items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # 指标
    payload = counters.to_dict()
    payload.update({
        'input_jsonl': input_path.as_posix(),
        'output_filtered_jsonl': output_filtered_path.as_posix(),
        'domain': domain,
    })
    write_metric(metric_file, 'classify_and_filter', payload)
    return counters


def main():
    parser = argparse.ArgumentParser(description='对 raw JSONL 进行四类判定并输出 filtered（仅情况1）。raw 原地覆盖添加 refuse/helpful/case 字段。')
    parser.add_argument('input_file', type=str, help='raw JSONL 文件路径（将被原地覆盖添加标注字段）')
    parser.add_argument('--output_filtered', type=str, required=True, help='过滤后的 JSONL 输出（仅情况1）')
    parser.add_argument('--metric_file', type=str, default=None)
    parser.add_argument('--domain', type=str, choices=['Animal', 'Bacteria'], default=None)
    parser.add_argument('--min_len', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=4096)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_filtered = Path(args.output_filtered)
    counters = process_one_file(input_path, output_filtered, args.metric_file, args.domain, args.min_len, args.max_len)

    print(f"完成: total={counters.total} case1={counters.num_case1} written={counters.num_written} extract_failed={counters.num_extract_failed}")


if __name__ == '__main__':
    main()
