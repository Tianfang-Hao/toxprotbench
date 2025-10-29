#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合各模型文件夹下的四个 TXT 小结，生成科研论文友好的 CSV。

扫描路径结构示例：
projects/Bio/benchmark/results/{DATE_TIME}/{MODEL_NAME}/benchmark_summary_{Domain}_temp-{T}.txt

输出 CSV 列包含：
- model, domain, temperature
- prompts, outputs
- case1_valid, case2_bad, case3_helpful_refusal, case4_unhelpful_refusal, parse_errors
- filter_total, filter_case1, filtered_written, extract_failed
- fasta_written, fasta_skipped
- classifier_name, classifier_rows, classifier_tox, classifier_non_tox, classifier_toxicity_rate
- esm_ppl_mean, esm_ppl_var (如果 TXT 中出现相关行则填充，否则留空)

使用：
python -m projects.Bio.benchmark.src.aggregate_benchmark_summaries \
  --results_root projects/Bio/benchmark/results \
  --out_csv projects/Bio/benchmark/results/benchmark_aggregate.csv
"""

import os
import re
import csv
import argparse
from typing import Dict, Any, Optional
import json

# --- 解析器 ---
SECTION_HEADER_RE = re.compile(r"^\[(?P<name>[^\]]+)\]\s*$")
KV_PAIR_RE = re.compile(r"(\w[\w_\-]*)\s*=\s*([^\s]+)")
TEMP_RE = re.compile(r"temp-([0-9]+\.?[0-9]*)")
DOMAIN_RE = re.compile(r"benchmark_summary_([A-Za-z]+)_temp-")


def parse_txt_summary(file_path: str) -> Dict[str, Any]:
    """解析单个 TXT 小结文件，返回 dict。尽量健壮，不抛异常。"""
    res: Dict[str, Any] = {}
    try:
        fname = os.path.basename(file_path)
        # 解析 domain 和 temperature
        m_dom = DOMAIN_RE.search(fname)
        if m_dom:
            res['domain'] = m_dom.group(1)
        else:
            # 兜底：尝试从路径中猜
            if 'Bacteria' in file_path:
                res['domain'] = 'Bacteria'
            elif 'Animal' in file_path:
                res['domain'] = 'Animal'
            else:
                res['domain'] = ''
        m_temp = TEMP_RE.search(fname)
        res['temperature'] = float(m_temp.group(1)) if m_temp else None

        current_section: Optional[str] = None
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = SECTION_HEADER_RE.match(line)
                if m:
                    current_section = m.group('name')
                    continue

                # 通用解析 key=value 对
                # 示例："prompts=200 outputs=200"
                # 或："total=200 case1_valid=8 case2_bad=192 ..."
                for k, v in KV_PAIR_RE.findall(line):
                    key = k.strip()
                    val = v.strip()
                    # 转为数值（int/float）
                    num: Any
                    if re.fullmatch(r"-?\d+", val):
                        num = int(val)
                    elif re.fullmatch(r"-?\d+\.\d+", val):
                        num = float(val)
                    else:
                        num = val

                    # 根据 section 前缀命名更语义化的键
                    if current_section == 'Generation':
                        if key in ('prompts', 'outputs'):
                            res[key] = num
                    elif current_section == 'Analyze Refuse':
                        # total, case1_valid, case2_bad, case3_helpful_refusal, case4_unhelpful_refusal, parse_errors
                        res[key] = num
                    elif current_section.startswith('Filter'):
                        # total, case1, written, extract_failed
                        prefix = 'filter_'
                        mapped = {
                            'total': 'filter_total',
                            'case1': 'filter_case1',
                            'written': 'filtered_written',
                            'extract_failed': 'extract_failed',
                        }
                        res[mapped.get(key, prefix + key)] = num
                    elif current_section == 'jsonl2fasta':
                        # written, skipped
                        res[f"fasta_{key}"] = num
                    else:
                        # 分类器段：可能是 [Bacteria Classifier] 或 [ToxinPred] 等
                        if current_section:
                            # 首次遇到分类器段时记下名字
                            if 'classifier_name' not in res:
                                res['classifier_name'] = current_section
                            # 试图识别常见键
                            name_map = {
                                'rows': 'classifier_rows',
                                'tox': 'classifier_tox',
                                'non-tox': 'classifier_non_tox',
                                'non_tox': 'classifier_non_tox',
                                'toxicity_rate': 'classifier_toxicity_rate',
                            }
                            out_key = name_map.get(key, f"{current_section.lower().replace(' ', '_')}_{key}")
                            res[out_key] = num

                # 可选：ESM PPL 行（若存在）
                # 例如："ppl_mean=3.21 ppl_var=1.02 n=123"
                if 'ppl_mean=' in line or 'ppl_var=' in line:
                    for k, v in KV_PAIR_RE.findall(line):
                        if k in ('ppl_mean', 'ppl_var'):
                            try:
                                res[f'esm_{k}'] = float(v)
                            except Exception:
                                pass
    except Exception:
        # 柔性失败：返回已解析的字段
        pass
    return res

def parse_json_summary(file_path: str) -> Dict[str, Any]:
    """解析 JSON 小结文件，返回统一字段命名的 dict。"""
    res: Dict[str, Any] = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        # domain / temperature
        res['domain'] = obj.get('domain', '')
        try:
            res['temperature'] = float(obj.get('temperature')) if obj.get('temperature') is not None else None
        except Exception:
            res['temperature'] = obj.get('temperature')

        # generation
        gen = obj.get('generation', {})
        if isinstance(gen, dict):
            if 'prompts' in gen:
                res['prompts'] = gen.get('prompts')
            if 'outputs' in gen:
                res['outputs'] = gen.get('outputs')

        # analyze_refuse
        arf = obj.get('analyze_refuse', {})
        if isinstance(arf, dict):
            for k in ('total','case1_valid','case2_bad','case3_helpful_refusal','case4_unhelpful_refusal','parse_errors'):
                if k in arf:
                    res[k] = arf.get(k)

        # classify_and_filter -> 规范为 filter_*
        caf = obj.get('classify_and_filter', {})
        if isinstance(caf, dict):
            if 'total' in caf:
                res['filter_total'] = caf.get('total')
            if 'num_case1' in caf:
                res['filter_case1'] = caf.get('num_case1')
            if 'num_written' in caf:
                res['filtered_written'] = caf.get('num_written')
            if 'num_extract_failed' in caf:
                res['extract_failed'] = caf.get('num_extract_failed')

        # jsonl2fasta
        j2f = obj.get('jsonl2fasta', {})
        if isinstance(j2f, dict):
            if 'written' in j2f:
                res['fasta_written'] = j2f.get('written')
            if 'skipped' in j2f:
                res['fasta_skipped'] = j2f.get('skipped')

        # esm ppl
        esm = obj.get('esm', {})
        if isinstance(esm, dict):
            if 'ppl_mean' in esm:
                res['esm_ppl_mean'] = esm.get('ppl_mean')
            if 'ppl_var' in esm:
                res['esm_ppl_var'] = esm.get('ppl_var')

        # classifier blocks
        bac = obj.get('bacteria_classifier', {})
        ani = obj.get('animal_toxinpred', {})
        cls_obj = None
        cls_name = None
        if isinstance(bac, dict) and bac:
            cls_obj = bac
            cls_name = 'Bacteria Classifier'
        elif isinstance(ani, dict) and ani:
            cls_obj = ani
            cls_name = 'ToxinPred'
        if cls_obj is not None:
            res['classifier_name'] = cls_name
            if 'rows' in cls_obj:
                res['classifier_rows'] = cls_obj.get('rows')
            # 统一 tox / non-tox 命名
            if 'toxic_rows' in cls_obj:
                res['classifier_tox'] = cls_obj.get('toxic_rows')
            if 'non_toxic_rows' in cls_obj:
                res['classifier_non_tox'] = cls_obj.get('non_toxic_rows')
            if 'toxicity_rate' in cls_obj:
                res['classifier_toxicity_rate'] = cls_obj.get('toxicity_rate')

    except Exception:
        pass
    return res


def infer_model_name_from_path(path: str) -> str:
    # 假设路径包含 .../{MODEL_NAME}/benchmark_summary_...
    parent = os.path.dirname(path)
    return os.path.basename(parent)


def collect_txt_summaries(results_root: str):
    """遍历 results_root，收集所有 benchmark_summary_*_temp-*.txt 文件。"""
    for root, dirs, files in os.walk(results_root):
        for fn in files:
            if not fn.startswith('benchmark_summary_'):
                continue
            if not fn.endswith('.txt'):
                continue
            if 'temp-' not in fn:
                continue
            yield os.path.join(root, fn)

def collect_json_summaries(results_root: str):
    """遍历 results_root，收集所有 benchmark_summary_*_temp-*.json 文件。"""
    for root, dirs, files in os.walk(results_root):
        for fn in files:
            if not fn.startswith('benchmark_summary_'):
                continue
            if not fn.endswith('.json'):
                continue
            if 'temp-' not in fn:
                continue
            yield os.path.join(root, fn)


def main():
    parser = argparse.ArgumentParser(description='聚合 benchmark TXT 小结为 CSV')
    parser.add_argument('--results_root', type=str, default='projects/Bio/benchmark/results', help='结果根目录（默认聚合所有日期/模型）')
    parser.add_argument('--run_dir', type=str, default=None, help='仅聚合某次运行目录，例如 results/2025-10-27_13-34-24')
    parser.add_argument('--model_dir', type=str, default=None, help='仅聚合某个模型目录，例如 results/<run>/<model>')
    parser.add_argument('--out_csv', type=str, default=None, help='输出 CSV 路径；若未指定则自动写到目标目录下 benchmark_aggregate.csv')
    args = parser.parse_args()

    # 确定扫描根目录
    scan_root = args.results_root
    scope = 'root'
    if args.model_dir:
        scan_root = args.model_dir
        scope = 'model'
    elif args.run_dir:
        scan_root = args.run_dir
        scope = 'run'

    # 默认输出路径
    out_csv = args.out_csv
    if not out_csv:
        if scope == 'model':
            out_csv = os.path.join(scan_root, 'benchmark_aggregate.csv')
        elif scope == 'run':
            out_csv = os.path.join(scan_root, 'benchmark_aggregate.csv')
        else:
            out_csv = os.path.join(scan_root, 'benchmark_aggregate.csv')

    def normalize(data: Dict[str, Any], model: str, path: str) -> Dict[str, Any]:
        return {
            'model': model,
            'domain': data.get('domain', ''),
            'temperature': data.get('temperature', ''),
            'prompts': data.get('prompts', ''),
            'outputs': data.get('outputs', ''),
            'case1_valid': data.get('case1_valid', ''),
            'case2_bad': data.get('case2_bad', ''),
            'case3_helpful_refusal': data.get('case3_helpful_refusal', ''),
            'case4_unhelpful_refusal': data.get('case4_unhelpful_refusal', ''),
            'parse_errors': data.get('parse_errors', ''),
            'filter_total': data.get('filter_total', ''),
            'filter_case1': data.get('filter_case1', ''),
            'filtered_written': data.get('filtered_written', ''),
            'extract_failed': data.get('extract_failed', ''),
            'fasta_written': data.get('fasta_written', ''),
            'fasta_skipped': data.get('fasta_skipped', ''),
            'classifier_name': data.get('classifier_name', ''),
            'classifier_rows': data.get('classifier_rows', ''),
            'classifier_tox': data.get('classifier_tox', ''),
            'classifier_non_tox': data.get('classifier_non_tox', ''),
            'classifier_toxicity_rate': data.get('classifier_toxicity_rate', ''),
            'esm_ppl_mean': data.get('esm_ppl_mean', ''),
            'esm_ppl_var': data.get('esm_ppl_var', ''),
            'source_path': path,
        }

    rows = []
    # 先收集 JSON，优先采用
    seen = set()
    for path in collect_json_summaries(scan_root):
        data = parse_json_summary(path)
        if not data:
            continue
        model_name = os.path.basename(os.path.dirname(path)) if scope == 'model' else infer_model_name_from_path(path)
        key = (model_name, str(data.get('domain','')), str(data.get('temperature','')))
        seen.add(key)
        rows.append(normalize(data, model_name, path))

    # 再补充 TXT 中那些没有对应 JSON 的项
    for path in collect_txt_summaries(scan_root):
        data = parse_txt_summary(path)
        if not data:
            continue
        model_name = os.path.basename(os.path.dirname(path)) if scope == 'model' else infer_model_name_from_path(path)
        key = (model_name, str(data.get('domain','')), str(data.get('temperature','')))
        if key in seen:
            continue
        rows.append(normalize(data, model_name, path))

    # 论文友好：排序顺序 model, domain, temperature
    rows.sort(key=lambda r: (r.get('model',''), r.get('domain',''), float(r.get('temperature') or 0)))

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    # 构建字段并确保“论文友好”的稳定顺序；对所有行取并集，防止首行缺列导致后续列被丢弃
    preferred_order = [
        'model','domain','temperature','prompts','outputs',
        'case1_valid','case2_bad','case3_helpful_refusal','case4_unhelpful_refusal','parse_errors',
        'filter_total','filter_case1','filtered_written','extract_failed',
        'fasta_written','fasta_skipped',
        'classifier_name','classifier_rows','classifier_tox','classifier_non_tox','classifier_toxicity_rate',
        'esm_ppl_mean','esm_ppl_var','source_path'
    ]
    all_keys = set(preferred_order)
    for r in rows:
        all_keys.update(r.keys())
    # 先按 preferred_order，剩余未知键按字母序追加
    fieldnames = [k for k in preferred_order if k in all_keys]
    tail_keys = sorted(all_keys - set(fieldnames))
    fieldnames.extend(tail_keys)

    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"已写出 CSV: {out_csv} (共 {len(rows)} 行)")


if __name__ == '__main__':
    main()
