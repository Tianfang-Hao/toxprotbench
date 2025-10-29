#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 hf-mirror 的匿名镜像加速从 Hugging Face 下载模型仓库（支持断点续传）。

特性：
- 默认将 HF_ENDPOINT 设置为 https://hf-mirror.com（可通过参数覆盖）。
- 可选启用 hf_transfer 以提速（若已安装 hf_transfer 则自动生效）。
- 支持仅下载权重分片与必要配置（--weights-only）。
- 自动对 5xx/超时做重试（指数退避）。
- 失败时给出可能原因（如仓库 gated 导致 403）。

示例：
  仅用镜像匿名下载完整仓库（推荐先确保磁盘充足）
    /data/nnotaa/miniconda3/envs/align/bin/python scripts/hf_mirror_snapshot_download.py \
      --repo-id Qwen/Qwen3-Next-80B-A3B-Instruct \
      --local-dir /data/nnotaa/align-anything/projects/Bio/benchmark/model/Qwen3-Next-80B-A3B-Instruct
        2>&1 | tee "/data/nnotaa/align-anything/projects/Bio/benchmark/model/download.log"
  仅下载权重与必要配置（流量更省）
    /data/nnotaa/miniconda3/envs/align/bin/python projects/Bio/benchmark/scripts/hf_mirror_snapshot_download.py \
      --repo-id Qwen/Qwen3-Next-80B-A3B-Instruct \
      --local-dir /data/models/Qwen3-Next-80B-A3B-Instruct \
      --weights-only
"""

import os
import sys
import time
import argparse
from typing import List, Optional

try:
    from huggingface_hub import snapshot_download
except Exception:
    print("错误：未找到 huggingface_hub，请先安装：\n  python -m pip install -U 'huggingface_hub[cli]'\n", file=sys.stderr)
    raise


def build_allow_patterns(weights_only: bool) -> Optional[list[str]]:
    if not weights_only:
        return None
    # 只拉权重与必要配置/分词器
    return [
        "*.safetensors",
        "config.json",
        "*.json",
        "*.model",
        "tokenizer.*",
        "*tokenizer.json",
        "merges.txt",
        "vocab.*",
        "*spiece.model",
    ]


def main():
    parser = argparse.ArgumentParser(description="hf-mirror 匿名镜像下载（支持断点续传+重试）")
    parser.add_argument("--repo-id", required=True, help="Hugging Face 仓库 ID，例如 Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--local-dir", required=True, help="下载落地目录")
    parser.add_argument("--endpoint", default="https://hf-mirror.com", help="HF 镜像端点（默认 https://hf-mirror.com）")
    parser.add_argument("--max-workers", type=int, default=6, help="最大并发下载数（稍小更稳）")
    parser.add_argument("--resume", action="store_true", default=True, help="启用断点续传（默认启用）")
    parser.add_argument("--no-symlinks", action="store_true", default=True, help="落地为真实文件而非符号链接（默认启用）")
    parser.add_argument("--weights-only", action="store_true", help="仅下载权重与必要配置/分词器")
    parser.add_argument("--print-env", action="store_true", help="打印关键环境变量用于排查")
    parser.add_argument("--retries", type=int, default=6, help="失败时重试次数（针对 5xx/超时/网关错误）")
    parser.add_argument("--backoff", type=float, default=6.0, help="重试初始等待秒数（指数退避）")
    args = parser.parse_args()

    # 配置镜像端点（若未外部指定，则本脚本设置）
    os.environ.setdefault("HF_ENDPOINT", args.endpoint)

    # 可选：启用 hf_transfer 加速（若已安装并愿意启用）
    transfer_on = False
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        transfer_on = True
    except Exception:
        pass

    if args.print_env:
        print("HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
        print("HF_HUB_ENABLE_HF_TRANSFER=", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))
        print("HUGGINGFACE_HUB_CACHE=", os.environ.get("HUGGINGFACE_HUB_CACHE"))

    allow_patterns = build_allow_patterns(args.weights_only)

    # 规范化布尔参数
    local_dir_use_symlinks = not bool(args.no_symlinks)
    resume_download = bool(args.resume)

    print("开始下载：")
    print("  repo_id=", args.repo_id)
    print("  local_dir=", args.local_dir)
    print("  endpoint=", os.environ.get("HF_ENDPOINT"))
    print("  hf_transfer=", transfer_on)
    print("  weights_only=", args.weights_only)

    attempt = 0
    while True:
        try:
            snapshot_download(
                repo_id=args.repo_id,
                local_dir=args.local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                resume_download=resume_download,
                max_workers=args.max_workers,
                allow_patterns=allow_patterns,
            )
            print("下载完成 ✅")
            print("提示：如需校验，可使用 `du -sh` 查看目录体量，或用 transformers 以 device_map=meta 加载配置进行干跑。")
            break
        except Exception as e:
            msg = str(e)
            print(f"下载失败（第 {attempt+1} 次）：", msg, file=sys.stderr)
            # 403/权限问题直接终止（大概率 gated）
            if "403" in msg or "Permission" in msg or "Access" in msg:
                print("可能原因：该仓库为 gated（需先同意许可）。匿名/无授权无法下载。", file=sys.stderr)
                print("可选方案：\n- 在可访问处登录 HF 同意许可后再下\n- 试试 ModelScope 是否有该模型并在那边同意许可\n- 或改用在线推理 API 暂时代替本地权重", file=sys.stderr)
                sys.exit(2)
            # 5xx/超时/网关错误：重试
            if attempt < args.retries - 1:
                wait = args.backoff * (2 ** attempt)
                wait = min(wait, 90.0)
                print(f"将在 {wait:.1f}s 后重试...", file=sys.stderr)
                time.sleep(wait)
                attempt += 1
                continue
            else:
                print("已达到最大重试次数，退出。", file=sys.stderr)
                sys.exit(3)


if __name__ == "__main__":
    main()
