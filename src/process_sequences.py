#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
import json
import csv
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time
import datetime
import sys 
# 假设 metrics.py 在同一目录或 PYTHONPATH 中
try:
    from metrics import write_metric
except ImportError:
    print("警告：无法导入 'metrics' 模块。'write_metric' 功能将不可用。", file=sys.stderr)
    def write_metric(path, name, data): pass # 定义一个空函数以避免崩溃

import threading

# --- 导入分布式计算所需模块 ---
import torch.distributed as dist
# DDP 已被移除

# --- 导入 Transformers 模块 ---
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

# --- 分布式设置函数 (保持不变) ---
def setup_distributed():
    """
    初始化分布式进程组。
    """
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    
    timeout = datetime.timedelta(minutes=30)
    
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout,
            device_id=local_rank,
        )
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)

def cleanup_distributed():
    """销毁分布式进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_sequence_id(data: dict) -> str:
    """从json行数据中获取一个唯一标识符 (保持不变)"""
    if 'id' in data: return str(data['id'])
    if 'prompt' in data: return str(data['prompt'])
    return data.get('assistant', '')


# --- 核心PPL计算 (保持不变) ---
def calculate_ppl_batch(
    sequences: list[str], 
    model: AutoModelForMaskedLM, 
    tokenizer: AutoTokenizer, 
    device: torch.device,
    heartbeat_cb=None,
    hb_interval: int = 50,
    pos_stride: int = 1, # 这个值现在将由 main 循环动态传入
) -> list[tuple[float, float, str]]: # 返回 (ppl, avg_log_likelihood, error_msg) 的列表
    """
    高性能批量计算PPL。
    一次模型调用处理一个批次中所有序列的同一个Token位置。
    """
    
    # 1. 准备和Tokenize批次
    sequences_spaced = [" ".join(list(s)) for s in sequences]
    try:
        inputs = tokenizer(
            sequences_spaced, 
            return_tensors='pt', 
            padding=True, 
            truncation=False
        ).to(device)
    except Exception as e:
        nan_result = (float('nan'), float('nan'), f"Tokenizer failed: {e}")
        return [nan_result] * len(sequences)

    token_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'] # [B, L]
    batch_size, padded_length = token_ids.shape

    # 2. 计算真实序列长度 (减去 [CLS] 和 [EOS])
    sequence_lengths = attention_mask.sum(dim=1) - 2
    sequence_lengths = torch.max(sequence_lengths, torch.tensor(1, device=device)) 
    
    max_len_in_batch = sequence_lengths.max().item()
    
    total_log_likelihoods = torch.zeros(batch_size, device=device, dtype=torch.float32)

    # 3. 逐个Token位置进行批量计算
    with torch.no_grad():
        # (*** 关键 ***)
        # pos_stride 现在由外部动态传入，以控制计算量
        stride = max(1, int(pos_stride))
        
        use_autocast = device.type == 'cuda'
        try:
            cap = torch.cuda.get_device_capability(device.index) if device.type == 'cuda' else (0, 0)
            amp_dtype = torch.bfloat16 if cap >= (8, 0) else torch.float16
        except Exception:
            amp_dtype = torch.float16

        # 预计算每个序列的采样位置计数（用于均值归一）
        sample_counts = (sequence_lengths + (stride - 1)) // stride
        sample_counts = torch.clamp(sample_counts, min=1)

        for i in range(0, max_len_in_batch, stride):
            pos = i + 1 # 当前Token位置 (跳过[CLS])
            
            original_token_ids_at_pos = token_ids[:, pos].clone()
            
            masked_token_ids = token_ids.clone()
            masked_token_ids[:, pos] = tokenizer.mask_token_id
            
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                    outputs = model(input_ids=masked_token_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=masked_token_ids, attention_mask=attention_mask)
            logits = outputs.logits # [B, L_padded, V]
            
            logits_at_pos = logits[:, pos, :] # [B, V]
            
            log_probs = torch.nn.functional.log_softmax(logits_at_pos, dim=-1)
            
            token_log_probs = torch.gather(
                log_probs, 
                dim=1, 
                index=original_token_ids_at_pos.unsqueeze(-1)
            ).squeeze(-1) # [B]
            
            is_real_token_mask = (i < sequence_lengths).float()
            
            total_log_likelihoods += token_log_probs * is_real_token_mask

            if heartbeat_cb is not None and (i % max(1, hb_interval) == 0):
                try:
                    heartbeat_cb()
                except Exception:
                    pass

    # 4. 计算最终PPL
    # 当 stride>1 时，使用采样计数归一（近似PLL）
    denom = sample_counts.float()
    avg_log_likelihoods = total_log_likelihoods / denom
    perplexities = torch.exp(-avg_log_likelihoods)

    # 5. 格式化输出
    results = []
    for j in range(batch_size):
        ppl = perplexities[j].item()
        avg_ll = avg_log_likelihoods[j].item()
        
        if math.isinf(ppl) or math.isnan(ppl):
            results.append((float('nan'), avg_ll, "PPL calculation resulted in inf/nan"))
        else:
            results.append((ppl, avg_ll, None))
            
    return results


# --- 动态批处理生成器 (保持不变) ---
def dynamic_batch_generator(lines: list[str], max_tokens_per_batch: int, solo_seq_len_threshold: int = 1024):
    """
    从行列表中按最大Token数生成批次，防止OOM。
    """
    batch_lines = []
    current_tokens = 0
    valid_chars = "ACDEFGHIKLMNPQRSTVWYX"

    for line in lines:
        try:
            data = json.loads(line.strip())
            sequence_raw = data.get('assistant', '')
            
            think_prefix = "<think>\n\n</think>\n\n"
            if sequence_raw.startswith(think_prefix):
                sequence_raw = sequence_raw.removeprefix(think_prefix)
            sequence_cleaned = sequence_raw.removesuffix('<|im_end|>')
            
            if not sequence_cleaned or not all(c.upper() in valid_chars for c in sequence_cleaned):
                yield [line] # 无效行，单独处理
                continue
            
            estimated_tokens = len(sequence_cleaned) + 2

            if len(sequence_cleaned) > solo_seq_len_threshold:
                if batch_lines:
                    yield batch_lines
                    batch_lines = []
                    current_tokens = 0
                yield [line]
                continue

            if not batch_lines or (current_tokens + estimated_tokens) <= max_tokens_per_batch:
                batch_lines.append(line)
                current_tokens += estimated_tokens
            else:
                yield batch_lines
                batch_lines = [line]
                current_tokens = estimated_tokens
                
        except (json.JSONDecodeError, KeyError, AttributeError):
            yield [line] # 格式错误的行，单独处理
            continue

    if batch_lines:
        yield batch_lines


def _estimate_seq_len_from_line(line: str, valid_chars: str = "ACDEFGHIKLMNPQRSTVWYX") -> int:
    """估算一行 JSON 对应序列的长度（非法或缺失返回0）。"""
    try:
        data = json.loads(line.strip())
        seq = str(data.get('assistant', '') or '')
        think_prefix = "<think>\n\n</think>\n\n"
        if seq.startswith(think_prefix):
            seq = seq.removeprefix(think_prefix)
        seq = seq.removesuffix('<|im_end|>')
        if not seq or not all(c.upper() in valid_chars for c in seq):
            return 0
        return len(seq)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="【分布式动态Stride版】处理文件夹中所有JSONL文件...")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--summary_csv_name", type=str, default="summary_results.csv")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t36_3B_UR50D")
    parser.add_argument("--max_tokens_per_batch", type=int, default=8192, help="每个GPU动态批次的最大Token总数，防OOM的关键。")
    parser.add_argument("--solo_seq_len_threshold", type=int, default=1024, help="序列长度超过该阈值将被单独成批，减少批内长尾。")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="可选：超过该长度的序列直接跳过并标记错误；0 表示不启用。(默认: 4096)")
    parser.add_argument("--max_sequences_per_task", type=int, default=4, help="每个任务文件最多包含的序列条数，减小任务粒度以提升负载均衡。")
    parser.add_argument("--task_lease_seconds", type=int, default=300, help="任务租约秒数：doing 目录下的任务若超过该时长未完成，将被回收并重新放回 todo。")
    parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")
    
    # --- (*** 关键修改 ***) ---
    parser.add_argument("--ppl_stride", type=int, default=1, 
                        help="动态步长的'侵略性乘数'。最终步长 = (L / BaseLen) * Multiplier。(默认: 1)")
    parser.add_argument("--stride_base_len", type=int, default=256, 
                        help="动态步长的'基准长度'。L <= BaseLen 时，stride=1*Multiplier。(默认: 256)")
    # --- 结束修改 ---

    parser.add_argument("--keep_intermediates", action="store_true", help="保留中间产物（默认会删除临时文件，如任务队列、心跳、临时分片等）")
    
    args = parser.parse_args()

    if not args.output_folder:
        args.output_folder = f"{args.input_folder.rstrip('/')}_esm"

    setup_distributed()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    def rank_print(msg):
        if rank == 0: print(msg)

    rank_print(f"动态Stride模式启动。共 {world_size} 个GPU。")
    rank_print(f"最大Tokens/批次: {args.max_tokens_per_batch}")
    rank_print(f"Stride 基准长度 (--stride_base_len): {args.stride_base_len}")
    rank_print(f"Stride 侵略性乘数 (--ppl_stride): {args.ppl_stride}")
    rank_print(f"最大序列长度 (--max_seq_len): {'无限制' if args.max_seq_len <= 0 else args.max_seq_len}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval() # 切换到评估模式

    rank_print(f"Rank {rank} 模型加载完成。")

    output_dir = Path(args.output_folder)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()

    # --- 文件发现 (Rank 0 执行) ---
    all_jsonl_files = []
    if rank == 0:
        input_dir = Path(args.input_folder)
        if input_dir.is_dir():
            all_jsonl_files = sorted([p for p in input_dir.glob("*.jsonl") if not p.name.endswith(".tmp")])
        else:
            rank_print(f"错误：输入文件夹 '{args.input_folder}' 不存在或不是一个目录。")
    
    file_list_container = [all_jsonl_files]
    dist.broadcast_object_list(file_list_container, src=0)
    all_jsonl_files = file_list_container[0]

    if not all_jsonl_files:
        rank_print("未找到 .jsonl 文件，程序退出。")
        cleanup_distributed()
        return

    file_progress_bar = tqdm(all_jsonl_files, desc="文件总进度", unit="file", disable=(rank != 0))
    valid_chars = "ACDEFGHIKLMNPQRSTVWYX" 

    total_lines = 0
    total_calculated = 0
    heartbeat_file = output_dir / f".rank_heartbeat.{rank}"
    status_file = output_dir / f".rank_status.{rank}"
    last_beat = time.time()
    
    # (*** 关键修改 ***) 提取基准长度，避免在循环中反复访问 args
    stride_base_len = max(1, args.stride_base_len)
    stride_multiplier = max(1, args.ppl_stride)

    for input_path in file_progress_bar:
        if rank == 0:
            file_progress_bar.set_postfix_str(f"处理中: {input_path.name}")
            
        # --- 断点续算 (所有GPU独立读取) ---
        processed_ids = set()
        output_path = output_dir / input_path.name
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f_out:
                for line in f_out:
                    try:
                        data = json.loads(line.strip())
                        if 'pseudo_perplexity' in data:
                            processed_ids.add(get_sequence_id(data))
                    except json.JSONDecodeError:
                        continue

        # --- 数据读取 (所有GPU独立读取) ---
        lines_to_process = []
        try:
            with open(input_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        data_id = get_sequence_id(json.loads(line.strip()))
                        if data_id not in processed_ids:
                            lines_to_process.append(line)
                    except json.JSONDecodeError:
                        lines_to_process.append(line) 
        except Exception as e:
            rank_print(f"严重错误：Rank {rank} 无法读取文件 {input_path.name}: {e}")
            continue

        # —— 按序列长度降序排序，以优先处理长序列，优化批处理填充 ——
        try:
            lines_to_process.sort(key=lambda ln: _estimate_seq_len_from_line(ln), reverse=True)
        except Exception:
            pass

        # --- 动态任务队列 ---
        task_root = output_dir / f".tasks_{input_path.name}"
        todo_dir = task_root / "todo"
        doing_dir = task_root / "doing"

        if rank == 0:
            # 准备任务目录（清理遗留）
            try:
                if task_root.exists():
                    for p in task_root.rglob('*'):
                        try: p.unlink()
                        except Exception: pass
                    try: task_root.rmdir()
                    except Exception: pass
                todo_dir.mkdir(parents=True, exist_ok=True)
                doing_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                rank_print(f"警告：无法准备任务目录 {task_root}: {e}")

            # --- (*** 关键修改 ***) 任务生成逻辑简化 ---
            # --- 不再区分长短序列，所有任务一视同仁 ---
            task_count = 0
            for batch_lines in dynamic_batch_generator(lines_to_process, args.max_tokens_per_batch, args.solo_seq_len_threshold):
                for i in range(0, len(batch_lines), max(1, args.max_sequences_per_task)):
                    sub_lines = batch_lines[i:i+max(1, args.max_sequences_per_task)]
                    # VVV 统一的任务名 VVV
                    task_path = todo_dir / f"{task_count:08d}.task" 
                    try:
                        with open(task_path, 'w', encoding='utf-8') as tf:
                            for ln in sub_lines:
                                tf.write(ln if ln.endswith('\n') else (ln + '\n'))
                        task_count += 1
                    except Exception as e:
                        rank_print(f"警告：写入任务失败 {task_path}: {e}")
            
            try:
                with open(task_root / 'READY', 'w') as rf:
                    rf.write(str(task_count))
            except Exception:
                pass

            total_chars = 0
            try:
                for ln in lines_to_process:
                    total_chars += _estimate_seq_len_from_line(ln)
            except Exception:
                total_chars = 0
            try:
                with open(task_root / 'TOTAL_CHARS', 'w') as rf:
                    rf.write(str(total_chars))
            except Exception:
                pass
            try:
                with open(task_root / 'PROGRESS_CHARS', 'w') as rf:
                    rf.write("0\n")
            except Exception:
                pass

        # 等待任务就绪
        try: dist.barrier()
        except Exception: pass

        temp_output_path = output_dir / f"{input_path.name}.rank_{rank}.tmp"
        
        monitor_thread = None
        if rank == 0:
            # 监控线程 (与之前版本相同，无需修改)
            def monitor_len_progress():
                try:
                    with open(task_root / 'TOTAL_CHARS', 'r') as f:
                        total = int(f.read().strip() or '0')
                except Exception:
                    total = 0
                bar = tqdm(total=total, desc="长度加权进度", unit="chars", leave=False) if total > 0 else None
                last = 0
                last_status_print = 0.0
                try:
                    while True:
                        if not task_root.exists(): break
                        try:
                            todo_list = list(todo_dir.glob('*.task'))
                            doing_list = list(doing_dir.glob('*.task*'))
                            has_todo = len(todo_list) > 0
                            has_doing = len(doing_list) > 0
                        except Exception:
                            has_todo = has_doing = False

                        processed = last
                        try:
                            with open(task_root / 'PROGRESS_CHARS', 'r') as pf:
                                vals = pf.read().strip().splitlines()
                                s = 0
                                for v in vals:
                                    try: s += int(v)
                                    except Exception: continue
                                processed = max(processed, s)
                        except Exception:
                            processed = last

                        if bar is not None and processed > last:
                            bar.update(processed - last)
                            last = processed

                        now_t = time.time()
                        if bar is not None and (now_t - last_status_print) > 2.0:
                            last_status_print = now_t
                            total_tasks = 0
                            try:
                                with open(task_root / 'READY', 'r') as rf:
                                    total_tasks = int((rf.read().strip() or '0'))
                            except Exception:
                                total_tasks = 0
                            todo_cnt = len(todo_list) if 'todo_list' in locals() else 0
                            doing_cnt = len(doing_list) if 'doing_list' in locals() else 0
                            done_cnt = max(total_tasks - todo_cnt - doing_cnt, 0)
                            try:
                                bar.set_postfix({
                                    'tasks': f"{done_cnt}/{total_tasks}",
                                    'todo': todo_cnt,
                                    'doing': doing_cnt,
                                })
                            except Exception:
                                pass
                            try:
                                statuses = []
                                for sp in sorted(output_dir.glob('.rank_status.*')):
                                    try:
                                        with open(sp, 'r', encoding='utf-8') as sf:
                                            info = json.loads(sf.read().strip() or '{}')
                                            info['_rank'] = sp.name.split('.')[-1]
                                            statuses.append(info)
                                    except Exception:
                                        continue
                                if statuses:
                                    statuses = sorted(statuses, key=lambda x: x.get('ts', 0), reverse=True)[:3]
                                    summary = []
                                    for st in statuses:
                                        # (*** 关键修改 ***) 新增 stride 状态显示
                                        summary.append(
                                            f"r{st.get('_rank','?')}: {st.get('file','?')} n={st.get('batch_size','?')} avgL={st.get('avg_len','?')} S={st.get('stride','?')}"
                                        )
                                    tqdm.write(" | ".join(summary))
                            except Exception:
                                pass

                        if not has_todo and not has_doing:
                            break
                        time.sleep(0.5)
                finally:
                    if bar is not None:
                        bar.close()
            
            monitor_thread = threading.Thread(target=monitor_len_progress, daemon=True)
            monitor_thread.start()
        
        with open(temp_output_path, 'w', encoding='utf-8') as temp_outfile:
            # 动态领取与处理任务
            while True:
                # 心跳
                now = time.time()
                if now - last_beat > 5:
                    try:
                        with open(heartbeat_file, 'w', encoding='utf-8') as hb:
                            hb.write(f"{rank} {now} polling {input_path.name}\n")
                    except Exception:
                        pass
                    last_beat = now

                # 获取一个任务
                candidate = None
                try:
                    # 排序的任务名（0000.task, 0001.task ...）
                    # 由于我们之前按长度降序排列，0000.task 总是最长的
                    todos = sorted([p for p in todo_dir.glob('*.task')])
                    if todos:
                        candidate = todos[0]
                except Exception:
                    candidate = None

                if candidate is None:
                    # 无待办：尝试回收过期 doing 任务（租约超时）
                    try:
                        now2 = time.time()
                        reclaimed = 0
                        for dp in sorted(doing_dir.glob('*.task*')):
                            try:
                                mtime = dp.stat().st_mtime
                            except Exception:
                                continue
                            if (now2 - mtime) > max(30, args.task_lease_seconds):
                                base_name = dp.name.split('.r', 1)[0] 
                                dest = todo_dir / base_name
                                if not dest.exists():
                                    try:
                                        os.rename(dp, dest)
                                        reclaimed += 1
                                    except Exception:
                                        pass
                        if reclaimed > 0:
                            continue
                    except Exception:
                        pass
                    
                    try:
                        still_doing = any(True for _ in doing_dir.glob('*.task*'))
                    except Exception:
                        still_doing = False
                    if not still_doing:
                        break
                    time.sleep(0.2)
                    continue

                # 试图原子领取（rename 原子）
                claim_path = doing_dir / f"{candidate.name}.r{rank}"
                try:
                    os.rename(candidate, claim_path)
                except FileNotFoundError:
                    continue
                except PermissionError:
                    continue
                except Exception:
                    continue

                # (*** 关键修改 ***) 移除任务名检查，因为所有任务都一样
                # task_filename = candidate.name
                # ... 移除 if task_filename.startswith(...) ...
                
                # 读取任务内容
                try:
                    with open(claim_path, 'r', encoding='utf-8') as tf:
                        batch_lines = tf.readlines()
                except Exception:
                    try: os.remove(claim_path)
                    except Exception: pass
                    continue

                # 处理该批次
                batch_data_to_write = []
                sequences_to_calc = []
                data_map = []
                for line in batch_lines:
                    try:
                        data = json.loads(line.strip())
                        if get_sequence_id(data) in processed_ids:
                            continue
                        sequence_raw = data.get('assistant', '')
                        think_prefix = "<think>\n\n</think>\n\n"
                        if sequence_raw.startswith(think_prefix):
                            sequence_raw = sequence_raw.removeprefix(think_prefix)
                        sequence_cleaned = sequence_raw.removesuffix('<|im_end|>')

                        # (*** 关键修改 ***) 使用 args.max_seq_len (默认 4096)
                        if args.max_seq_len > 0 and len(sequence_cleaned) > args.max_seq_len:
                            data['pseudo_perplexity'] = None
                            data['error'] = f"Sequence too long (> {args.max_seq_len}), skipped"
                            batch_data_to_write.append(data)
                            total_lines += 1
                            continue

                        if sequence_cleaned and all(c.upper() in valid_chars for c in sequence_cleaned):
                            sequences_to_calc.append(sequence_cleaned)
                            data_map.append(data)
                        else:
                            data['pseudo_perplexity'] = None
                            data['error'] = "Invalid or empty sequence"
                            batch_data_to_write.append(data)
                        total_lines += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        error_data = {'original_line': line.strip(), 'error': f'JSON format error: {e}'}
                        batch_data_to_write.append(error_data)

                if sequences_to_calc:
                    try:
                        # (*** 关键修改 ***) 动态计算 Stride
                        current_max_len = 0
                        try:
                            current_max_len = max(len(s) for s in sequences_to_calc)
                        except ValueError:
                            pass # 批次为空
                        
                        # 1. 计算基础步长
                        base_dynamic_stride = (current_max_len + stride_base_len - 1) // stride_base_len
                        # 2. 应用侵略性乘数
                        current_stride = max(1, base_dynamic_stride * stride_multiplier)
                        
                        # 写入状态文件
                        try:
                            ids_sample = [get_sequence_id(d) for d in data_map[:2]]
                            avg_len = int(sum(len(s) for s in sequences_to_calc) / max(1, len(sequences_to_calc)))
                            status_payload = {
                                'ts': time.time(),
                                'file': input_path.name,
                                'batch_size': len(sequences_to_calc),
                                'avg_len': avg_len,
                                'ids_sample': ids_sample,
                                'stride': current_stride # VVV 显示当前stride VVV
                            }
                            with open(status_file, 'w', encoding='utf-8') as sf:
                                sf.write(json.dumps(status_payload, ensure_ascii=False))
                        except Exception:
                            pass
                        
                        def hb():
                            try:
                                with open(heartbeat_file, 'w', encoding='utf-8') as hb_f:
                                    hb_f.write(f"{rank} {time.time()} batch_hb {input_path.name}\n")
                            except Exception:
                                pass
                        
                        # (*** 关键修改 ***) 使用动态决定的 current_stride
                        ppl_results = calculate_ppl_batch(
                            sequences_to_calc,
                            model,
                            tokenizer,
                            device,
                            heartbeat_cb=hb,
                            hb_interval=50,
                            pos_stride=current_stride,
                        )
                        
                        try:
                            batch_chars = sum(len(s) for s in sequences_to_calc)
                            with open(task_root / 'PROGRESS_CHARS', 'a') as pf:
                                pf.write(str(batch_chars) + "\n")
                        except Exception:
                            pass

                        for i, (ppl, avg_ll, err_msg) in enumerate(ppl_results):
                            data = data_map[i]
                            data['pseudo_perplexity'] = ppl if not math.isnan(ppl) else None
                            data['avg_log_likelihood'] = avg_ll if not math.isnan(avg_ll) else None
                            if err_msg:
                                data['error'] = err_msg
                            batch_data_to_write.append(data)
                            if data['pseudo_perplexity'] is not None:
                                total_calculated += 1
                    except Exception as e:
                        print(f"Rank {rank} FATAL batch error (ID: {get_sequence_id(data_map[0])}): {e}", file=sys.stderr)
                        for data in data_map:
                            data['pseudo_perplexity'] = None
                            data['error'] = f"Batch calculation fatal error: {str(e)}"
                            batch_data_to_write.append(data)

                # 写入本批次结果
                for data in batch_data_to_write:
                    temp_outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                # 释放任务
                try: os.remove(claim_path)
                except Exception: pass

        # 完成一个文件也更新一次心跳
        try:
            with open(heartbeat_file, 'w', encoding='utf-8') as hb:
                hb.write(f"{rank} {time.time()} done_file {input_path.name}\n")
        except Exception:
            pass

        # 仅 rank0 清理任务目录
        if rank == 0:
            try:
                if monitor_thread is not None:
                    monitor_thread.join(timeout=2)
            except Exception:
                pass
            if not args.keep_intermediates: # 仅在不保留时才清理
                try:
                    for p in task_root.rglob('*'):
                        try: p.unlink(missing_ok=True)
                        except Exception: pass
                    task_root.rmdir()
                except Exception:
                    pass

    # --- 文件合并与摘要生成 (使用文件同步回退) ---
    rank_print("\n所有进程计算完成，写入 done 文件并等待其他进程（文件同步回退）...")

    try:
        done_file = output_dir / f".rank_done.{rank}"
        with open(done_file, 'w', encoding='utf-8') as f:
            f.write(f"{rank}\n")
    except Exception as e:
        rank_print(f"警告: 无法写入 done 文件 {done_file}: {e}")

    wait_timeout = 120 
    poll_interval = 1.0
    start_wait = time.time()

    if rank == 0:
        rank_print("开始等待所有 rank 的 done 文件...")
        total_chars_all = 0
        try:
            # (*** 关键修改 ***) 修复：如果 keep_intermediates=True，任务目录仍在
            for p in output_dir.glob('.tasks_*'):
                tp = p / 'TOTAL_CHARS'
                if tp.exists():
                    with open(tp, 'r') as f:
                        v = int(f.read().strip() or '0')
                        total_chars_all += max(0, v)
        except Exception:
            total_chars_all = 0

        wait_bar = tqdm(total=world_size, desc="等待GPU完成", unit="rank", leave=False)
        len_bar = tqdm(total=total_chars_all, desc="长度加权进度", unit="chars", leave=False) if total_chars_all > 0 else None
        last_postfix = ""
        while True:
            now = time.time()
            try:
                done_files = list(output_dir.glob('.rank_done.*'))
            except Exception:
                done_files = []

            try:
                present = set()
                for p in done_files:
                    try:
                        present.add(int(p.name.split('.')[-1]))
                    except Exception:
                        continue
            except Exception:
                present = set()

            missing = [r for r in range(world_size) if r not in present]
            num_done = len(present)
            
            if wait_bar.n != num_done:
                wait_bar.n = num_done
                wait_bar.refresh()

            oldest_heartbeat_age = None
            try:
                ages = []
                for r in missing:
                    hb_file = output_dir / f".rank_heartbeat.{r}"
                    if hb_file.exists():
                        try:
                            with open(hb_file, 'r', encoding='utf-8') as hb:
                                parts = hb.read().strip().split()
                                if len(parts) >= 2:
                                    ts = float(parts[1])
                                    ages.append(max(0, now - ts))
                        except Exception:
                            continue
                if ages:
                    oldest_heartbeat_age = int(max(ages))
            except Exception:
                oldest_heartbeat_age = None

            postfix = f"done={num_done}/{world_size}"
            if missing: postfix += f" missing={missing}"
            if oldest_heartbeat_age is not None: postfix += f" oldest_hb={oldest_heartbeat_age}s"
            
            if postfix != last_postfix:
                wait_bar.set_postfix_str(postfix)
                last_postfix = postfix

            if num_done >= world_size:
                rank_print(f"已检测到所有 {world_size} 个 done 文件。")
                break
            if time.time() - start_wait > wait_timeout:
                if missing:
                    rank_print(f"等待 done 文件超时 ({wait_timeout}s)。缺失 rank: {missing}。继续合并已有临时文件。")
                else:
                    rank_print(f"等待 done 文件超时 ({wait_timeout}s)，将继续合并已有临时文件。")
                break
            time.sleep(poll_interval)
        
        try: wait_bar.close()
        except Exception: pass
        try: 
            if len_bar is not None: len_bar.close()
        except Exception: pass

        rank_print("开始合并临时文件...")
        for input_path in tqdm(all_jsonl_files, desc="合并文件", unit="file"):
            output_path = output_dir / input_path.name
            
            original_lines_data = []
            try:
                with open(input_path, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        if line.strip():
                            try:
                                original_lines_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                original_lines_data.append({'original_line': line.strip(), 'error': 'Original JSON format error'})
            except Exception as e:
                rank_print(f"严重错误：无法读取原始输入文件 {input_path.name} 进行合并: {e}")
                continue

            processed_results = {}
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f_old_out:
                    for line in f_old_out:
                        try:
                            data = json.loads(line)
                            if 'pseudo_perplexity' in data: 
                                processed_results[get_sequence_id(data)] = data
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            new_results = {}
            for r in range(world_size):
                temp_path = output_dir / f"{input_path.name}.rank_{r}.tmp"
                if temp_path.exists():
                    with open(temp_path, 'r', encoding='utf-8') as temp_infile:
                        for line in temp_infile:
                            try:
                                data = json.loads(line)
                                new_results[get_sequence_id(data)] = data
                            except (json.JSONDecodeError, KeyError):
                                continue
                    if not args.keep_intermediates: # 不保留才删除
                        try: os.remove(temp_path)
                        except Exception: pass

            final_results = {**processed_results, **new_results}

            with open(output_path, 'w', encoding='utf-8') as outfile:
                for original_data in original_lines_data:
                    seq_id = get_sequence_id(original_data)
                    result_data = final_results.get(seq_id, original_data) 
                    outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    if rank == 0:
        try:
            merge_file = output_dir / '.merge_complete'
            with open(merge_file, 'w', encoding='utf-8') as f:
                f.write(f"merged_by_rank0 {time.time()}\n")
        except Exception as e:
            rank_print(f"警告: 无法写入 merge_complete: {e}")

    start_wait2 = time.time()
    while True:
        try:
            if (output_dir / '.merge_complete').exists():
                break
        except Exception:
            pass
        if time.time() - start_wait2 > wait_timeout:
            rank_print(f"等待 merge_complete 超时 ({wait_timeout}s)，继续后续步骤。")
            break
        time.sleep(poll_interval)

    # --- CSV 摘要 (不变) ---
    if rank == 0:
        rank_print("正在生成最终的CSV摘要文件...")
        final_summaries = []
        
        for output_path in tqdm(sorted(list(output_dir.glob("*.jsonl"))), desc="生成摘要", unit="file"):
            # 仅处理常规文件，跳过任务队列目录等
            try:
                if output_path.name.startswith('.'):  # 隐藏/内部标记
                    continue
                if not output_path.is_file():
                    continue
            except Exception:
                continue
            perplexities = []
            line_count = 0
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    try:
                        data = json.loads(line.strip())
                        ppl = data.get('pseudo_perplexity')
                        if ppl is not None:
                            perplexities.append(ppl)
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            avg_ppl = (sum(perplexities) / len(perplexities)) if perplexities else float('nan')
            final_summaries.append({
                'source_file': output_path.name,
                'average_perplexity': avg_ppl,
                'sequence_count': line_count,
                'calculated_count': len(perplexities)
            })

        if final_summaries:
            summary_csv_path = output_dir / args.summary_csv_name
            try:
                with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['source_file', 'average_perplexity', 'sequence_count', 'calculated_count']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(final_summaries)
                print(f"摘要文件成功保存到: {summary_csv_path}")
            except IOError as e:
                print(f"错误：无法写入摘要文件 {summary_csv_path}。原因: {e}")
        else:
            print("处理完成，但没有生成任何有效的摘要信息。")

    # 写指标：仅 rank 0 汇总
    if rank == 0:
        write_metric(args.metric_file, "process_sequences_esm", {
            "input_dir": args.input_folder,
            "output_dir": output_dir.as_posix(),
            "input_files": len(all_jsonl_files),
            "num_read": total_lines,
            "num_calculated": total_calculated,
            "num_failed": max(total_lines - total_calculated, 0),
            "model_name": args.model_name,
            "summary_csv": (output_dir / args.summary_csv_name).as_posix(),
        })

        # 最终清理
        if not args.keep_intermediates:
            rank_print("正在清理中间文件...")
            try:
                # 1) rank 临时合并分片
                for p in output_dir.glob('*.rank_*.tmp'):
                    try: p.unlink(missing_ok=True)
                    except Exception: pass
                # 2) 心跳与完成标记
                for p in output_dir.glob('.rank_heartbeat.*'):
                    try: p.unlink(missing_ok=True)
                    except Exception: pass
                for p in output_dir.glob('.rank_status.*'):
                    try: p.unlink(missing_ok=True)
                    except Exception: pass
                for p in output_dir.glob('.rank_done.*'):
                    try: p.unlink(missing_ok=True)
                    except Exception: pass
                # 3) 合并完成标记
                try: (output_dir / '.merge_complete').unlink(missing_ok=True)
                except Exception: pass
                # 4) 动态任务队列目录
                for d in output_dir.glob('.tasks_*'):
                    try:
                        for sub in sorted(d.rglob('*'), reverse=True):
                            try: sub.unlink(missing_ok=True)
                            except IsADirectoryError:
                                try: sub.rmdir()
                                except Exception: pass
                            except Exception: pass
                        try: d.rmdir()
                        except Exception: pass
                    except Exception: pass
            except Exception as e:
                rank_print(f"清理中间文件时出错: {e}")

    # ---- 每个 rank 清理自己的文件 ----
    try: (output_dir / f".rank_heartbeat.{rank}").unlink(missing_ok=True)
    except Exception: pass
    try: (output_dir / f".rank_status.{rank}").unlink(missing_ok=True)
    except Exception: pass
    try: (output_dir / f".rank_done.{rank}").unlink(missing_ok=True)
    except Exception: pass

    # 仅 rank 0 额外清理遗留文件
    if rank == 0:
        try: (output_dir / '.merge_complete').unlink(missing_ok=True)
        except Exception: pass
        # 扫描删除可能遗留的文件
        try:
            for p in output_dir.glob('.rank_heartbeat.*'):
                try: p.unlink(missing_ok=True)
                except Exception: continue
            for p in output_dir.glob('.rank_done.*'):
                try: p.unlink(missing_ok=True)
                except Exception: continue
            for p in output_dir.glob('.rank_status.*'):
                try: p.unlink(missing_ok=True)
                except Exception: continue
        except Exception:
            pass

    cleanup_distributed()


if __name__ == '__main__':
    main()