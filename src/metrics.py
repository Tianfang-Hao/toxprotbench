#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的结构化指标工具。

提供两个函数：
- write_metric(metric_file, step, data)
- with metric_timer(metric_file, step, extra=None): ...

约定：
- 每条指标以 JSONL 形式追加写入 metric_file。
- JSON 字段至少包含：step, timestamp, duration_sec（如有）, 以及 data 中的其他键值。
- 任何写入错误都不会抛出，避免影响主流程。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _ensure_parent_dir(path: str) -> None:
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
    except Exception:
        # 安静失败，不影响主流程
        pass


def write_metric(metric_file: Optional[str], step: str, data: Dict[str, Any]) -> None:
    """
    安全写入一条指标记录到 JSONL 文件。

    参数：
        metric_file: 指标文件路径；为 None 或空则不执行。
        step: 阶段名，例如 'filter_no_imend'
        data: 附加数据字典，将合并到基础字段中。
    """
    if not metric_file:
        return
    try:
        _ensure_parent_dir(metric_file)
        payload = {
            "step": step,
            "timestamp": float(time.time()),
        }
        # 合并数据（后者覆盖前者）
        if isinstance(data, dict):
            payload.update(data)
        with open(metric_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 指标写入失败时静默
        pass


@contextmanager
def metric_timer(metric_file: Optional[str], step: str, extra: Optional[Dict[str, Any]] = None):
    """
    用于度量一个代码块的执行耗时，并自动写入一条 metric。

    用法：
        with metric_timer(metric_file, "filter_no_imend", {"input_files": 3}):
            ...
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        data = {"duration_sec": float(duration)}
        if extra:
            data.update(extra)
        write_metric(metric_file, step, data)
