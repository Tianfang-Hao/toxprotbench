#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取批量推理生成的 JSONL 文件并进行后处理：
1) 如果 assistant 不是“单独的一段蛋白质序列”，尝试从文本中提取一段氨基酸序列；若提取成功，则替换 assistant 为该序列。
2) 如果无法提取到氨基酸序列，则将该条目标注为 refuse:false, helpful:true（视为未拒绝但具有帮助性）。

默认允许的氨基酸字符为 20 种标准氨基酸字母加 X：ACDEFGHIKLMNPQRSTVWYX。
"""

import argparse
import json
import re
import sys
import os
from typing import Tuple, Optional
from metrics import write_metric

# 允许的氨基酸字符（大写）
AA_SET = set("ACDEFGHIKLMNPQRSTVWYX")


IM_END_TOKEN = "<|im_end|>"


def is_single_sequence(text: str, min_len: int) -> bool:
    """判断文本是否就是一段单独的氨基酸序列（只接受大写 AA_SET 与空白）。"""
    if not isinstance(text, str):
        return False
    # 去除空白与换行，但不改变大小写
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < min_len:
        return False
    # 只接受大写字母集合中的字符
    return all(ch in AA_SET for ch in cleaned)


def extract_sequence(text: str, min_len: int) -> Optional[str]:
    """从文本中提取最长的一段由 AA_SET(大写) 组成的序列，长度需 >= min_len；若找不到则返回 None。"""
    if not isinstance(text, str) or not text:
        return None
    # 仅匹配大写氨基酸字母的连续段（不做大小写转换）
    pattern = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{" + str(min_len) + r",}")
    matches = pattern.findall(text)
    if not matches:
        return None
    # 返回最长的匹配段
    return max(matches, key=len)


def process_file(
    input_path: str,
    metric_file: Optional[str],
    min_len: int,
    append_imend: bool = True,
    drop_nonseq: bool = True,
    require_finish_stop: bool = False,
    output_path: Optional[str] = None,
) -> Tuple[int, int, int, int, int, int]:
    """处理单个 JSONL 文件，原地覆盖写回。

    返回统计：总数、已是单序列数、提取成功数、标注 helpful 数、追加 im_end 次数、丢弃条数。
    """
    total = 0
    already_single = 0
    extracted = 0
    marked_helpful = 0
    appended_imend = 0

    items = []
    kept_items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # 跳过非法行
                continue

    for obj in items:
        total += 1
        assistant = obj.get('assistant', '')
        # 可选：先依据 finish_reason 过滤（仅当参数启用）
        if require_finish_stop and obj.get('finish_reason') != 'stop':
            # 直接丢弃
            continue
        # 标准化：移除空白，但不改变大小写；仅当原本已是大写序列才视为单序列
        if isinstance(assistant, str):
            normalized = re.sub(r"\s+", "", assistant)
        else:
            normalized = ""

        if is_single_sequence(assistant, min_len=min_len):
            # 已是单序列：标准化并可选追加 im_end
            already_single += 1
            if append_imend:
                obj['assistant'] = normalized + IM_END_TOKEN
                appended_imend += 1
            else:
                obj['assistant'] = normalized
            obj['extracted'] = False
            kept_items.append(obj)
            continue

        seq = extract_sequence(assistant, min_len=min_len)
        if seq:
            # 提取到序列：用提取的序列替换 assistant，保留一个标记字段
            if append_imend:
                obj['assistant'] = seq + IM_END_TOKEN
                appended_imend += 1
            else:
                obj['assistant'] = seq
            obj['extracted'] = True
            extracted += 1
            kept_items.append(obj)
        else:
            # 未提取到：默认丢弃以清理纯自然语言；若显式禁用则回落到标注 helpful
            if drop_nonseq:
                # 丢弃，不计入 kept_items
                pass
            else:
                obj['refuse'] = False
                obj['helpful'] = True
                obj['extracted'] = False
                marked_helpful += 1
                kept_items.append(obj)

    # 写出到 output_path（若未指定则覆盖 input_path，保持向后兼容）
    target = output_path or input_path
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, 'w', encoding='utf-8') as f:
        for obj in kept_items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # 写指标
    dropped = max(total - (already_single + extracted + marked_helpful), 0)
    write_metric(metric_file, 'extract_or_mark_helpful', {
        'input_jsonl': input_path,
        'total': total,
        'already_single': already_single,
        'extracted': extracted,
        'marked_helpful': marked_helpful,
        'min_len': min_len,
        'appended_imend': appended_imend,
        'dropped': dropped,
        'drop_nonseq': drop_nonseq,
        'require_finish_stop': require_finish_stop,
    })

    return total, already_single, extracted, marked_helpful, appended_imend, dropped


def main():
    parser = argparse.ArgumentParser(description='从 JSONL 的 assistant 文本中提取序列，默认丢弃无序列的纯自然语言，并可选追加 <|im_end|> 与按 finish_reason 过滤。')
    parser.add_argument('input_file', type=str, help='输入 JSONL 文件路径（原地覆盖写回）')
    parser.add_argument('--metric_file', type=str, default=None, help='指标输出 JSONL 文件')
    parser.add_argument('--min_len', type=int, default=5, help='判定/提取序列的最小长度')
    parser.add_argument('--no_append_imend', action='store_true', help='不追加 <|im_end|> 标记（默认会追加）')
    parser.add_argument('--keep_nonseq', action='store_true', help='保留未能提取序列的行，并标注 helpful（默认丢弃）')
    parser.add_argument('--require_finish_stop', action='store_true', help='仅保留 finish_reason=="stop" 的行（若字段存在）')
    parser.add_argument('--output_file', type=str, default=None, help='将处理结果写入指定文件（而不是就地覆盖）')
    args = parser.parse_args()

    total, already_single, extracted, marked_helpful, appended_imend, dropped = process_file(
        input_path=args.input_file,
        metric_file=args.metric_file,
        min_len=args.min_len,
        append_imend=(not args.no_append_imend),
        drop_nonseq=(not args.keep_nonseq),
        require_finish_stop=args.require_finish_stop,
        output_path=args.output_file,
    )

    print(f"处理完成: 共{total}条，已是单序列{already_single}，提取成功{extracted}，标注 helpful {marked_helpful}，丢弃 {dropped}，追加 im_end {appended_imend}。")


if __name__ == '__main__':
    main()
