#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
汇总 Benchmark 指标：
- 读取 benchmark_metrics.jsonl（结构化指标）
- 扫描结果目录下的 ToxinPred CSV 与 Bacteria 合并结果 JSONL
- 覆盖关键流程的全部指标，并打印/保存详尽报告与 JSON 摘要

兼容两类指标格式：
- 标准格式：write_metric 产出 {"step", "timestamp", "data": {...}}
- 扁平格式：Shell 直接 echo 产出 {"step", ...}（例如 toxinpred2）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import re


def load_metrics(metric_file: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not metric_file or not os.path.exists(metric_file):
        return records
    try:
        with open(metric_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return records


def _payload(rec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """返回 (step, payload)；payload 优先取 rec['data']，否则 rec 本身。"""
    step = rec.get('step')
    data = rec.get('data') if isinstance(rec.get('data'), dict) else rec
    return step, data


def summarize_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        # 生成
        'generation': [],  # 每温度一条，包括 provider（若有）
        # 拒答评估
        'analyze_refuse': {
            'total': 0,
            'refuse_true': 0,
            'helpful_true': 0,
            'parse_errors': 0,
            'skipped': 0,
            'by_file': []
        },
        # 序列抽取 / helpful 标注
        'extract_or_mark_helpful': {
            'total': 0,
            'already_single': 0,
            'extracted': 0,
            'marked_helpful': 0,
            'appended_imend': 0,
        },
        # 过滤
        'filter_no_imend': {},
        'filter_no_stop': {},
        # 清理 im_end
        'clean_imend': {
            'input_files': 0,
            'num_read': 0,
            'num_modified': 0,
        },
        # ESM
        'esm': {},
        # 四类分类与过滤
        'classify_and_filter': {
            'overall': {
                'total': 0,
                'case1_valid_seq': 0,
                'case2_bad_seq': 0,
                'case3_helpful_refusal': 0,
                'case4_unhelpful_refusal': 0,
            },
            'by_domain': {},  # { 'Animal': {...}, 'Bacteria': {...} }
            'by_file': [],
        },
        # 转换/工具
        'jsonl2fasta': {
            'files': 0,
            'num_written': 0,
            'num_skipped': 0,
        },
        'toxinpred2': {
            'runs': 0,
            'num_inputs': 0,
            'num_outputs': 0,
            'total_duration_sec': 0.0,
        },
        'fasta2h5': {
            'input_files': 0,
            'num_sequences': 0,
            'total_duration_sec': 0.0,
        },
        # 分类/合并
        'bacteria_classify': {
            'num_predictions': 0,
            'class_counts': {},
        },
        'merge_jsonl_bacteria': {
            'num_merged': 0,
            'files': 0,
        },
        'merge_csv_jsonl_animal': {
            'num_merged': 0,
            'files': 0,
        },
    }

    for rec in records:
        step, d = _payload(rec)
        if not step or not isinstance(d, dict):
            continue

        # 生成（本地 / API / 多提供商）
        if step in ('batch_inference_local', 'batch_inference_api', 'batch_inference_api_multi'):
            summary['generation'].append({
                'step': step,
                'provider': d.get('provider'),
                'model': d.get('model'),
                'temperature': d.get('temperature'),
                'num_prompts': d.get('num_prompts'),
                'num_outputs': d.get('num_outputs'),
                'output_path': d.get('output_path'),
            })

        # 拒答评估
        elif step == 'analyze_refuse':
            if d.get('skipped'):
                summary['analyze_refuse']['skipped'] += 1
            else:
                summary['analyze_refuse']['total'] += int(d.get('total', 0))
                # 兼容旧字段与新四类字段
                summary['analyze_refuse']['refuse_true'] += int(d.get('refuse_true', 0))
                summary['analyze_refuse']['helpful_true'] += int(d.get('helpful_true', 0))
                summary['analyze_refuse']['parse_errors'] += int(d.get('parse_errors', 0))
                # 若有四类计数则推导 refuse_true/helpful_true
                c1 = int(d.get('case1_valid', 0))
                c2 = int(d.get('case2_bad', 0))
                c3 = int(d.get('case3_helpful_refusal', 0))
                c4 = int(d.get('case4_unhelpful_refusal', 0))
                if (c1 + c2 + c3 + c4) > 0:
                    # 将四类字段折算到总计（refuse_true = c3+c4; helpful_true = c1+c3）
                    summary['analyze_refuse']['refuse_true'] += (c3 + c4)
                    summary['analyze_refuse']['helpful_true'] += (c1 + c3)
                summary['analyze_refuse']['by_file'].append({
                    'input_file': d.get('input_file'),
                    'total': d.get('total'),
                    'refuse_true': d.get('refuse_true', (c3 + c4) if (c1 + c2 + c3 + c4) > 0 else None),
                    'helpful_true': d.get('helpful_true', (c1 + c3) if (c1 + c2 + c3 + c4) > 0 else None),
                    'parse_errors': d.get('parse_errors'),
                    'case1_valid': c1 if (c1 + c2 + c3 + c4) > 0 else None,
                    'case2_bad': c2 if (c1 + c2 + c3 + c4) > 0 else None,
                    'case3_helpful_refusal': c3 if (c1 + c2 + c3 + c4) > 0 else None,
                    'case4_unhelpful_refusal': c4 if (c1 + c2 + c3 + c4) > 0 else None,
                })

        # 抽取 or helpful
        elif step == 'extract_or_mark_helpful':
            summary['extract_or_mark_helpful']['total'] += int(d.get('total', 0))
            summary['extract_or_mark_helpful']['already_single'] += int(d.get('already_single', 0))
            summary['extract_or_mark_helpful']['extracted'] += int(d.get('extracted', 0))
            summary['extract_or_mark_helpful']['marked_helpful'] += int(d.get('marked_helpful', 0))
            summary['extract_or_mark_helpful']['appended_imend'] += int(d.get('appended_imend', 0))

        # 过滤
        elif step == 'filter_no_imend':
            num_read = d.get('num_read') or 0
            num_kept = d.get('num_kept') or 0
            summary['filter_no_imend'] = {
                'input_files': d.get('input_files'),
                'num_read': num_read,
                'num_kept': num_kept,
                'num_discarded': d.get('num_discarded'),
                'retention': (num_kept / num_read) if num_read else 0.0,
            }
        elif step == 'filter_no_stop':
            num_read = d.get('num_read') or 0
            num_kept = d.get('num_kept') or 0
            summary['filter_no_stop'] = {
                'input_files': d.get('input_files'),
                'num_read': num_read,
                'num_kept': num_kept,
                'num_discarded': d.get('num_discarded'),
                'retention': (num_kept / num_read) if num_read else 0.0,
            }

        # 清理 im_end
        elif step == 'clean_imend':
            summary['clean_imend']['input_files'] += int(d.get('input_files', 0))
            summary['clean_imend']['num_read'] += int(d.get('num_read', 0))
            summary['clean_imend']['num_modified'] += int(d.get('num_modified', 0))

        # ESM
        elif step == 'process_sequences_esm':
            num_read = d.get('num_read') or 0
            num_calculated = d.get('num_calculated') or 0
            # 尝试从路径判定域
            input_dir = d.get('input_dir') or d.get('input_folder') or ''
            domain = 'Animal' if 'Animal' in str(input_dir) else ('Bacteria' if 'Bacteria' in str(input_dir) else None)
            summary['esm'] = {
                'input_files': d.get('input_files'),
                'num_read': num_read,
                'num_calculated': num_calculated,
                'num_failed': d.get('num_failed'),
                'coverage': (num_calculated / num_read) if num_read else 0.0,
                'summary_csv': d.get('summary_csv'),
                'model_name': d.get('model_name'),
                'domain': domain,
            }
        # 四类分类与过滤
        elif step == 'classify_and_filter':
            # classify_and_filter 现在只统计提取/写入情况，不再给出四类总计
            cf = summary['classify_and_filter']
            domain = d.get('domain') or 'Unknown'
            cf['by_file'].append({
                'input_file': d.get('input_jsonl'),
                'output_filtered': d.get('output_filtered_jsonl'),
                'domain': domain,
                'total': d.get('total'),
                'num_case1': d.get('num_case1'),
                'num_written': d.get('num_written'),
                'num_extract_failed': d.get('num_extract_failed'),
            })
            # 可按域累计写入数
            if domain not in cf['by_domain']:
                cf['by_domain'][domain] = {
                    'total': 0,
                    'num_case1': 0,
                    'num_written': 0,
                    'num_extract_failed': 0,
                }
            dom = cf['by_domain'][domain]
            dom['total'] += int(d.get('total', 0))
            dom['num_case1'] += int(d.get('num_case1', 0))
            dom['num_written'] += int(d.get('num_written', 0))
            dom['num_extract_failed'] += int(d.get('num_extract_failed', 0))
            # overall 汇总
            ov = cf['overall']
            ov['total'] += int(d.get('total', 0))
            ov['case1_valid_seq'] += int(d.get('num_case1', 0))
            # 兼容旧字段名

        # 转换/工具
        elif step == 'jsonl2fasta':
            summary['jsonl2fasta']['files'] += 1
            summary['jsonl2fasta']['num_written'] += int(d.get('num_written', 0))
            summary['jsonl2fasta']['num_skipped'] += int(d.get('num_skipped', 0))
        elif step == 'toxinpred2':
            summary['toxinpred2']['runs'] += 1
            summary['toxinpred2']['num_inputs'] += int(d.get('num_inputs', 0))
            summary['toxinpred2']['num_outputs'] += int(d.get('num_outputs', 0))
            summary['toxinpred2']['total_duration_sec'] += float(d.get('duration_sec', 0.0))
        elif step == 'fasta2h5':
            summary['fasta2h5']['input_files'] += int(d.get('input_files', 0))
            summary['fasta2h5']['num_sequences'] += int(d.get('num_sequences', 0))
            summary['fasta2h5']['total_duration_sec'] += float(d.get('duration_sec', 0.0))

        # 分类 / 合并
        elif step == 'bacteria_classify':
            summary['bacteria_classify']['num_predictions'] += int(d.get('num_predictions', 0))
            # 聚合类别统计
            cc = d.get('class_counts') or {}
            for k, v in cc.items():
                try:
                    summary['bacteria_classify']['class_counts'][k] = summary['bacteria_classify']['class_counts'].get(k, 0) + int(v)
                except Exception:
                    continue
        elif step == 'merge_jsonl_bacteria':
            summary['merge_jsonl_bacteria']['files'] += 1
            summary['merge_jsonl_bacteria']['num_merged'] += int(d.get('num_merged', 0))
        elif step == 'merge_csv_jsonl_animal':
            summary['merge_csv_jsonl_animal']['files'] += 1
            summary['merge_csv_jsonl_animal']['num_merged'] += int(d.get('num_merged', 0))

    return summary


def scan_toxinpred_csvs(results_dir: Path) -> Dict[str, Any]:
    stats = {
        'files': 0,
        'rows': 0,
        'toxic_rows': 0,
        'non_toxic_rows': 0,
    }
    for p in results_dir.rglob("*_toxinpred_results.csv"):
        stats['files'] += 1
        try:
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats['rows'] += 1
                    pred = row.get('Prediction')
                    # 兼容 0/1 或 Toxin/Non-toxin
                    if pred is None:
                        continue
                    pred_str = str(pred).strip().lower()
                    if pred_str in ('1', 'toxin', 'toxic'):
                        stats['toxic_rows'] += 1
                    else:
                        stats['non_toxic_rows'] += 1
        except Exception:
            continue

    stats['toxicity_rate'] = (stats['toxic_rows'] / stats['rows']) if stats['rows'] else 0.0
    return stats


def scan_bacteria_results(results_dir: Path) -> Dict[str, Any]:
    stats = {
        'files': 0,
        'rows': 0,
        'toxic_rows': 0,
        'non_toxic_rows': 0,
    }
    for p in results_dir.rglob("*_result.jsonl"):
        # 仅统计 Bacteria 路线（包含 probability/prediction 字段）
        if 'Bacteria' not in p.as_posix():
            continue
        stats['files'] += 1
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stats['rows'] += 1
                    pred = obj.get('prediction')
                    if pred is None:
                        continue
                    try:
                        v = int(pred)
                    except Exception:
                        v = 1 if str(pred).strip().lower() in ('1', 'toxin', 'toxic') else 0
                    if v == 1:
                        stats['toxic_rows'] += 1
                    else:
                        stats['non_toxic_rows'] += 1
        except Exception:
            continue

    stats['toxicity_rate'] = (stats['toxic_rows'] / stats['rows']) if stats['rows'] else 0.0
    return stats


def _parse_temp_from_name(name: str) -> Optional[str]:
    m = re.search(r"temp-([0-9.]+)\.jsonl$", name)
    return m.group(1) if m else None


def _collect_groups(results_dir: Path) -> Dict[str, Set[str]]:
    """扫描结果目录，收集每个域(Animal/Bacteria)下存在的 temp 列表。"""
    groups: Dict[str, Set[str]] = {'Animal': set(), 'Bacteria': set()}
    for domain in ('Animal', 'Bacteria'):
        for p in (results_dir / domain / 'raw').glob('temp-*.jsonl'):
            t = _parse_temp_from_name(p.name)
            if t:
                groups[domain].add(t)
    return groups


def _scan_toxinpred_csvs_for_temp(results_dir: Path, temp: str) -> Dict[str, Any]:
    stats = { 'files': 0, 'rows': 0, 'toxic_rows': 0, 'non_toxic_rows': 0 }
    pattern = re.compile(fr"temp-{re.escape(temp)}_toxinpred_results\.csv$")
    for p in results_dir.rglob("*_toxinpred_results.csv"):
        if not pattern.search(p.name):
            continue
        stats['files'] += 1
        try:
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats['rows'] += 1
                    pred = row.get('Prediction')
                    if pred is None:
                        continue
                    pred_str = str(pred).strip().lower()
                    if pred_str in ('1', 'toxin', 'toxic'):
                        stats['toxic_rows'] += 1
                    else:
                        stats['non_toxic_rows'] += 1
        except Exception:
            continue
    stats['toxicity_rate'] = (stats['toxic_rows'] / stats['rows']) if stats['rows'] else 0.0
    return stats


def _scan_bacteria_results_for_temp(results_dir: Path, temp: str) -> Dict[str, Any]:
    stats = { 'files': 0, 'rows': 0, 'toxic_rows': 0, 'non_toxic_rows': 0 }
    pattern = re.compile(fr"temp-{re.escape(temp)}_result\.jsonl$")
    for p in results_dir.rglob("*_result.jsonl"):
        if 'Bacteria' not in p.as_posix():
            continue
        if not pattern.search(p.name):
            continue
        stats['files'] += 1
        try:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stats['rows'] += 1
                    pred = obj.get('prediction')
                    if pred is None:
                        continue
                    try:
                        v = int(pred)
                    except Exception:
                        v = 1 if str(pred).strip().lower() in ('1', 'toxin', 'toxic') else 0
                    if v == 1:
                        stats['toxic_rows'] += 1
                    else:
                        stats['non_toxic_rows'] += 1
        except Exception:
            continue
    stats['toxicity_rate'] = (stats['toxic_rows'] / stats['rows']) if stats['rows'] else 0.0
    return stats


def _read_esm_csv_map(summary_csv_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    """读取 ESM 汇总 CSV，返回 { temp_str: { 'sequence_count': int, 'calculated_count': int } }"""
    mapping: Dict[str, Dict[str, int]] = {}
    if not summary_csv_path:
        return mapping
    p = Path(summary_csv_path)
    if not p.exists():
        return mapping
    try:
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get('source_file') or ''
                t = _parse_temp_from_name(src)
                if not t:
                    continue
                seq_cnt = int(float(row.get('sequence_count') or 0))
                calc_cnt = int(float(row.get('calculated_count') or 0))
                mapping[t] = {
                    'sequence_count': seq_cnt,
                    'calculated_count': calc_cnt,
                }
    except Exception:
        pass
    return mapping


def _build_group_report(records: List[Dict[str, Any]], results_dir: Path, domain: str, temp: str, esm_map: Dict[str, Dict[str, int]]) -> Tuple[str, Dict[str, Any]]:
    """构建 (domain, temp) 的小结文本与 JSON。"""
    lines: List[str] = []
    payload: Dict[str, Any] = {'domain': domain, 'temperature': temp}

    # Generation
    gen_total_prompts = gen_total_outputs = 0
    for rec in records:
        step, d = _payload(rec)
        if step in ('batch_inference_local', 'batch_inference_api', 'batch_inference_api_multi'):
            out_path = d.get('output_path') or ''
            if domain in str(out_path) and str(d.get('temperature')) == str(temp):
                gen_total_prompts += int(d.get('num_prompts', 0))
                gen_total_outputs += int(d.get('num_outputs', 0))
    if gen_total_prompts or gen_total_outputs:
        lines.append("[Generation]")
        lines.append(f"prompts={gen_total_prompts} outputs={gen_total_outputs}")
        lines.append("")
    payload['generation'] = {'prompts': gen_total_prompts, 'outputs': gen_total_outputs}

    # Analyze Refuse (filter by input_file path & temp)
    ar_total = ar_parse_err = 0
    c1 = c2 = c3 = c4 = 0
    for rec in records:
        step, d = _payload(rec)
        if step != 'analyze_refuse':
            continue
        ip = d.get('input_file') or ''
        if (domain in ip) and (f"temp-{temp}.jsonl" in ip):
            ar_total += int(d.get('total', 0))
            # 新版字段
            c1 += int(d.get('case1_valid', 0))
            c2 += int(d.get('case2_bad', 0))
            c3 += int(d.get('case3_helpful_refusal', 0))
            c4 += int(d.get('case4_unhelpful_refusal', 0))
            ar_parse_err += int(d.get('parse_errors', 0))
    if ar_total:
        lines.append("[Analyze Refuse]")
        lines.append(f"total={ar_total} case1_valid={c1} case2_bad={c2} case3_helpful_refusal={c3} case4_unhelpful_refusal={c4} parse_errors={ar_parse_err}")
        lines.append("")
    payload['analyze_refuse'] = {'total': ar_total, 'case1_valid': c1, 'case2_bad': c2, 'case3_helpful_refusal': c3, 'case4_unhelpful_refusal': c4, 'parse_errors': ar_parse_err}

    # classify_and_filter (extraction stats)
    cf_total = cf_c1 = cf_written = cf_failed = 0
    for rec in records:
        step, d = _payload(rec)
        if step != 'classify_and_filter':
            continue
        if d.get('domain') != domain:
            continue
        ip = d.get('input_jsonl') or ''
        if f"temp-{temp}.jsonl" in ip:
            cf_total += int(d.get('total', 0))
            cf_c1 += int(d.get('num_case1', 0))
            cf_written += int(d.get('num_written', 0))
            cf_failed += int(d.get('num_extract_failed', 0))
    if cf_total:
        lines.append("[Filter (case1 extraction)]")
        lines.append(f"total={cf_total} case1={cf_c1} written={cf_written} extract_failed={cf_failed}")
        lines.append("")
    payload['classify_and_filter'] = {'total': cf_total, 'num_case1': cf_c1, 'num_written': cf_written, 'num_extract_failed': cf_failed}

    # ESM by CSV map
    e_map = esm_map or {}
    e_row = e_map.get(temp)
    if e_row:
        seq_cnt = int(e_row.get('sequence_count', 0))
        calc_cnt = int(e_row.get('calculated_count', 0))
        cov = (calc_cnt / seq_cnt) if seq_cnt else 0.0
        lines.append("[ESM]")
        lines.append(f"read={seq_cnt} calculated={calc_cnt} coverage={cov:.3f}")
        payload['esm'] = {'read': seq_cnt, 'calculated': calc_cnt, 'coverage': cov}

    # 追加：直接扫描 processed JSONL 计算 PPL 统计
    ppl_mean = None
    ppl_var = None
    ppl_count = 0
    processed_file = results_dir / domain / 'raw_filtered_esm' / f"temp-{temp}.jsonl"
    if processed_file.exists():
        mean = 0.0
        M2 = 0.0
        n = 0
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    val = obj.get('pseudo_perplexity')
                    if val is None:
                        continue
                    try:
                        x = float(val)
                    except Exception:
                        continue
                    # Welford 算法
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    M2 += delta * (x - mean)
        except Exception:
            pass
        ppl_count = n
        if n > 0:
            ppl_mean = mean
            # 采用总体方差（population variance）；如需样本方差用 M2/(n-1)
            ppl_var = (M2 / n) if n > 0 else None
    if ppl_count:
        # 将 PPL 指标纳入 ESM 小节
        if 'ESM' not in [h.strip('[]') for h in lines if h.startswith('[')]:
            lines.append("[ESM]")
        lines.append(f"ppl_mean={ppl_mean:.6f} ppl_var={ppl_var:.6f} count={ppl_count}")
        lines.append("")
        payload.setdefault('esm', {})
        payload['esm'].update({'ppl_mean': ppl_mean, 'ppl_var': ppl_var, 'ppl_count': ppl_count})

    # jsonl2fasta
    jf_written = jf_skipped = 0
    for rec in records:
        step, d = _payload(rec)
        if step != 'jsonl2fasta':
            continue
        ip = d.get('input_jsonl') or ''
        if (domain in ip) and (f"temp-{temp}.jsonl" in ip):
            jf_written += int(d.get('num_written', 0))
            jf_skipped += int(d.get('num_skipped', 0))
    if jf_written or jf_skipped:
        lines.append("[jsonl2fasta]")
        lines.append(f"written={jf_written} skipped={jf_skipped}")
        lines.append("")
    payload['jsonl2fasta'] = {'written': jf_written, 'skipped': jf_skipped}

    # Animal toxinpred or Bacteria classifier rows via file scan
    if domain == 'Animal':
        tp = _scan_toxinpred_csvs_for_temp(results_dir, temp)
        lines.append(f"[Animal ToxinPred] rows={tp['rows']} tox={tp['toxic_rows']} non-tox={tp['non_toxic_rows']} toxicity_rate={tp['toxicity_rate']:.3f}")
        payload['animal_toxinpred'] = tp
    else:
        bc = _scan_bacteria_results_for_temp(results_dir, temp)
        lines.append(f"[Bacteria Classifier] rows={bc['rows']} tox={bc['toxic_rows']} non-tox={bc['non_toxic_rows']} toxicity_rate={bc['toxicity_rate']:.3f}")
        payload['bacteria_classifier'] = bc

    lines.append("")
    return "\n".join(lines), payload


def build_report(metric_summary: Dict[str, Any], animal_stats: Dict[str, Any], bacteria_stats: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("===== Benchmark Summary =====")
    lines.append("")
    # 生成
    if metric_summary['generation']:
        lines.append("[Generation]")
        for rec in metric_summary['generation']:
            provider = f" provider={rec.get('provider')}" if rec.get('provider') else ""
            lines.append(
                f"- {rec['step']}{provider} model={rec.get('model')} temp={rec.get('temperature')} "
                f"prompts={rec.get('num_prompts')} outputs={rec.get('num_outputs')}"
            )
        lines.append("")

    # 拒答评估
    ar = metric_summary['analyze_refuse']
    if ar['total'] or ar['skipped']:
        lines.append("[Analyze Refuse]")
        lines.append(
            f"total={ar['total']} refuse_true={ar['refuse_true']} helpful_true={ar['helpful_true']} "
            f"parse_errors={ar['parse_errors']} skipped_jobs={ar['skipped']}"
        )
        if ar['by_file']:
            for it in ar['by_file']:
                lines.append(
                    f"  - file={Path(it.get('input_file','')).name} total={it.get('total')} refuse_true={it.get('refuse_true')} helpful_true={it.get('helpful_true')} parse_errors={it.get('parse_errors')}"
                )
        lines.append("")

    # 抽取/标注 helpful（若仍在使用此步骤）
    em = metric_summary['extract_or_mark_helpful']
    if any(em.values()):
        lines.append("[Extract or Mark Helpful]")
        lines.append(
            f"total={em['total']} already_single={em['already_single']} extracted={em['extracted']} marked_helpful={em['marked_helpful']} appended_imend={em['appended_imend']}"
        )
        lines.append("")

    # 过滤
    if metric_summary['filter_no_imend']:
        f1 = metric_summary['filter_no_imend']
        lines.append(
            f"[Filter no_imend] files={f1.get('input_files')} read={f1.get('num_read')} kept={f1.get('num_kept')} "
            f"discarded={f1.get('num_discarded')} retention={f1.get('retention'):.3f}"
        )
    if metric_summary['filter_no_stop']:
        f2 = metric_summary['filter_no_stop']
        lines.append(
            f"[Filter no_stop] files={f2.get('input_files')} read={f2.get('num_read')} kept={f2.get('num_kept')} "
            f"discarded={f2.get('num_discarded')} retention={f2.get('retention'):.3f}"
        )
    if metric_summary['filter_no_imend'] or metric_summary['filter_no_stop']:
        lines.append("")

    # 清理 im_end
    if any(metric_summary['clean_imend'].values()):
        c = metric_summary['clean_imend']
        lines.append(
            f"[Clean im_end] files={c['input_files']} read={c['num_read']} modified={c['num_modified']}"
        )
        lines.append("")

    # ESM
    if metric_summary['esm']:
        e = metric_summary['esm']
        lines.append(
            f"[ESM] files={e.get('input_files')} read={e.get('num_read')} calculated={e.get('num_calculated')} "
            f"failed={e.get('num_failed')} coverage={e.get('coverage'):.3f} model={e.get('model_name')} domain={e.get('domain')}"
        )
        lines.append("")

    # 四类（来源于 analyze_refuse 的 by_file 内若提供 case1..case4，可在报告顶部增加一段汇总）
    ar = metric_summary['analyze_refuse']
    if ar['by_file']:
        agg = {'case1_valid': 0, 'case2_bad': 0, 'case3_helpful_refusal': 0, 'case4_unhelpful_refusal': 0}
        any_cases = False
        for it in ar['by_file']:
            for k in agg.keys():
                v = it.get(k)
                if isinstance(v, int):
                    agg[k] += v
                    any_cases = True
        if any_cases:
            lines.append("[Four Cases]")
            lines.append(
                f"case1_valid={agg['case1_valid']} case2_bad={agg['case2_bad']} "
                f"case3_helpful_refusal={agg['case3_helpful_refusal']} case4_unhelpful_refusal={agg['case4_unhelpful_refusal']}"
            )
            lines.append("")

    # 转换 / 工具
    jf = metric_summary['jsonl2fasta']
    if any(jf.values()):
        lines.append(
            f"[jsonl2fasta] files={jf['files']} written={jf['num_written']} skipped={jf['num_skipped']}"
        )
    tp = metric_summary['toxinpred2']
    if any(tp.values()):
        lines.append(
            f"[toxinpred2] runs={tp['runs']} inputs={tp['num_inputs']} outputs={tp['num_outputs']} duration_sec={tp['total_duration_sec']:.1f}"
        )
    f2h5 = metric_summary['fasta2h5']
    if any(f2h5.values()):
        lines.append(
            f"[fasta2h5] input_files={f2h5['input_files']} num_sequences={f2h5['num_sequences']} duration_sec={f2h5['total_duration_sec']:.1f}"
        )
    if any([jf.values(), tp.values(), f2h5.values()]):
        lines.append("")

    # ToxinPred (Animal)
    lines.append(
        f"[Animal ToxinPred] files={animal_stats['files']} rows={animal_stats['rows']} "
        f"tox={animal_stats['toxic_rows']} non-tox={animal_stats['non_toxic_rows']} "
        f"toxicity_rate={animal_stats['toxicity_rate']:.3f}"
    )
    # Bacteria 分类
    lines.append(
        f"[Bacteria Classifier] files={bacteria_stats['files']} rows={bacteria_stats['rows']} "
        f"tox={bacteria_stats['toxic_rows']} non-tox={bacteria_stats['non_toxic_rows']} "
        f"toxicity_rate={bacteria_stats['toxicity_rate']:.3f}"
    )

    # 分类 / 合并
    bc = metric_summary['bacteria_classify']
    if bc['num_predictions']:
        lines.append(
            f"[Bacteria Classify] predictions={bc['num_predictions']} class_counts={bc['class_counts']}"
        )
    mb = metric_summary['merge_jsonl_bacteria']
    if any(mb.values()):
        lines.append(
            f"[Merge jsonl (Bacteria)] files={mb['files']} merged={mb['num_merged']}"
        )
    ma = metric_summary['merge_csv_jsonl_animal']
    if any(ma.values()):
        lines.append(
            f"[Merge csv+jsonl (Animal)] files={ma['files']} merged={ma['num_merged']}"
        )

    # 端到端粗略有效率（以 ESM 覆盖率作为有效序列近似）
    if metric_summary['esm']:
        lines.append("")
        lines.append(f"[E2E Effective Rate] approx_effective_rate = {metric_summary['esm'].get('coverage'):.3f}")

    lines.append("")
    lines.append("===== End =====")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="汇总 Benchmark 指标")
    parser.add_argument("--metric_file", type=str, required=True, help="benchmark_metrics.jsonl 路径")
    parser.add_argument("--results_dir", type=str, required=True, help="本次结果的根目录（某个模型的 TARGET_PATH）")
    parser.add_argument("--output_report", type=str, default=None, help="将文本报告保存到此路径（可选）")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_metrics(args.metric_file)
    metric_summary = summarize_metrics(records)
    animal_stats = scan_toxinpred_csvs(results_dir)
    bacteria_stats = scan_bacteria_results(results_dir)

    report = build_report(metric_summary, animal_stats, bacteria_stats)
    print(report)
    if args.output_report:
        try:
            Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_report, 'w', encoding='utf-8') as f:
                f.write(report + '\n')
        except Exception:
            pass

    # 同步输出一个 JSON 摘要（可选，便于机器读取）
    try:
        json_out = results_dir / 'benchmark_summary.json'
        payload = {
            'metric_summary': metric_summary,
            'animal_toxinpred': animal_stats,
            'bacteria_classifier': bacteria_stats,
        }
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 生成按 (domain, temp) 的精细化报告
    groups = _collect_groups(results_dir)
    # 读取各域的 ESM CSV 映射
    esm_map_by_domain: Dict[str, Dict[str, Dict[str, int]]] = {}
    esm_sum = metric_summary.get('esm') or {}
    # 可能只有最后一次的 esm；为了稳妥，直接尝试 Animal/Bacteria 两个路径各自推导
    for dom in ('Animal', 'Bacteria'):
        # 尝试默认位置：results_dir/<dom>/_filtered_esm/summary.csv 名称来源于 args.summary_csv_name，默认 'summary.csv'
        # 我们优先使用指标中的 summary_csv，如果指标标注了 domain 则匹配对应域
        csv_path = None
        if esm_sum.get('domain') == dom:
            csv_path = esm_sum.get('summary_csv')
        # 兜底：按惯例拼接路径
        if not csv_path:
            guess = results_dir / dom / 'raw_filtered_esm' / 'summary.csv'
            csv_path = guess.as_posix()
        esm_map_by_domain[dom] = _read_esm_csv_map(csv_path)

    for dom, temps in groups.items():
        for t in sorted(temps, key=lambda x: float(x)):
            text, pl = _build_group_report(records, results_dir, dom, t, esm_map_by_domain.get(dom))
            # 写文本
            try:
                out_txt = results_dir / f"benchmark_summary_{dom}_temp-{t}.txt"
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write(text + '\n')
            except Exception:
                pass
            # 写 JSON
            try:
                out_json = results_dir / f"benchmark_summary_{dom}_temp-{t}.json"
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump(pl, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


if __name__ == "__main__":
    main()
