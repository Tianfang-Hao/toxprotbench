#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5 模型批量分布式推理脚本 (命令行版本)

功能:
- 支持从 JSONL 文件中读取多个 prompt (及其关联数据) 进行批量推理。
- [已修改] 生成的 JSONL 文件会保留原始 JSONL 的所有字段，并新增一个 "assistant" 字段。
- 通过命令行参数支持对多个本地模型和多个温度参数进行组合测试。
- 使用 PyTorch Distributed Data Parallel (DDP) 在多 GPU 环境下进行分布式推理，显著提升效率。
- 为每个模型和温度的组合生成独立的 JSONL 结果文件。
- 提供详细的进度条来监控整个推理过程。

使用方法:
    1.  确保已安装所有依赖库。
    2.  在终端中使用 `torchrun` 命令并附带所需参数来启动脚本。

示例命令:
    # 使用 2 个 GPU 对两个模型、三个温度进行推理
    torchrun --nproc_per_node=2 batch_inference_cli_updated.py \
        --prompt_file_path "/path/to/your/prompts.jsonl" \
        --prompt_key "prompt_text" \
        --output_dir "/path/to/save/results" \
        --model_paths "/path/to/Qwen2.5-7B-Instruct/slice_2690" "/path/to/Another-Model/checkpoint-1000" \
        --temperatures 0.0 0.7 1.0 \
        --max_new_tokens 256 \
        --top_p 0.8 \
        --repetition_penalty 1.0

依赖库:
    - transformers, torch, accelerate, sentencepiece
    - tqdm (用于显示进度条)

作者: 生成自 align-anything 框架 (由 Gemini 修改)
日期: 2025-09-12 (更新于 2025-09-13)
"""

# 导入必要的库
import os
import json
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from metrics import write_metric
from datetime import datetime
import argparse
import threading
import time
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Qwen 模型批量分布式推理脚本 (命令行版本)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 文件 I/O 配置 ---
    parser.add_argument(
        "--prompt_file_path",
        type=str,
        required=True,
        help="输入的 prompt 文件路径 (JSONL 格式)。"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="generated_prompt_for_sequence_model",
        help="JSONL 文件中包含 prompt 文本的键 (key)。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出结果的根目录。"
    )

    # --- 模型与推理配置 ---
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs='+',
        required=True,
        help="需要进行推理的一个或多个模型路径，用空格分隔。"
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs='+',
        required=True,
        help="需要测试的一个或多个温度值，用空格分隔。"
    )

    # --- 生成参数配置 ---
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="生成的最大 token 数量。"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) 采样参数。"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="重复惩罚系数。"
    )
    parser.add_argument(
        "--metric_file",
        type=str,
        default=None,
        help="结构化指标输出文件 (JSONL)，可选。"
    )
    
    return parser.parse_args()


def setup_distributed():
    """初始化分布式环境"""
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        # 如果环境不支持分布式或只有一个 GPU，则返回单机配置
        return 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def load_data(file_path, prompt_key):
    """
    [已修改] 从 JSONL 文件加载所有数据 (字典列表)
    """
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 确保 prompt_key 存在
                    if prompt_key not in item:
                        raise KeyError(f"键 '{prompt_key}' 不在行中")
                    data_list.append(item)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告: 跳过格式错误或缺少键 '{prompt_key}' 的行: {line.strip()} - 错误: {e}")
    except FileNotFoundError:
        print(f"错误: Prompt 文件未找到 -> {file_path}")
        return None
    return data_list

def get_model_response(model, tokenizer, user_input, generation_config):
    """根据单个用户输入，生成模型回复"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input + '\nAttention: This is all the information. Please provide the most likely protein sequence. The answer should only contain the single-letter codes for the 20 amino acids that make up the protein. Do not use three-letter codes or include any other characters, including spaces.' }
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]

    generation_kwargs = generation_config.copy()
    generation_kwargs['pad_token_id'] = tokenizer.eos_token_id
    
    outputs = model.generate(**model_inputs, **generation_kwargs)
    response_ids = outputs[0][input_length:]
    
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False).strip()
    return response_text

def run_inference_on_rank(rank, world_size, model_path, temp, data_on_this_rank, base_gen_config, prompt_key):
    """
    [已修改] 单个 rank (GPU) 上的推理执行函数
    """
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(rank)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 配置当前温度的生成参数
    current_gen_config = base_gen_config.copy()
    if temp == 0.0:
        current_gen_config["do_sample"] = False
        current_gen_config.pop("temperature", None)
        current_gen_config.pop("top_p", None)
    else:
        current_gen_config["do_sample"] = True
        current_gen_config["temperature"] = temp

    # 为当前 rank 上的 prompts 进行推理
    results = []
    progress_bar = tqdm(
        data_on_this_rank, # [修改] 迭代字典列表
        desc=f"GPU-{rank} Infer", 
        disable=(rank != 0),
        position=1,
        leave=False
    )
    completed = 0
    # 轻量进度文件（跨 rank 汇总用），rank 写入，rank0 汇总展示
    progress_dir = None
    progress_file = None
    try:
        # 仅当 output_dir 可用时在外层赋值，这里延迟到主函数传参会更好，但为最小改动暂用环境变量传递
        progress_dir = os.environ.get("AA_LOCAL_PROGRESS_DIR", None)
        if progress_dir:
            Path(progress_dir).mkdir(parents=True, exist_ok=True)
            progress_file = os.path.join(progress_dir, f"rank_{rank}.progress")
            with open(progress_file, 'w') as pf:
                pf.write("0\n")
    except Exception:
        progress_file = None

    for item in progress_bar:
        # [修改] 复制原始字典，以保留所有字段
        result_item = item.copy()
        
        # [修改] 从字典中提取 prompt
        prompt = result_item.get(prompt_key)
        
        if prompt is None:
             # 理论上 load_data 已经检查过了，但作为安全措施
            print(f"警告: GPU-{rank} 发现缺少 prompt_key '{prompt_key}' 的项目，跳过。")
            continue
        
        response = get_model_response(model, tokenizer, prompt, current_gen_config)
        
        # [修改] 将 assistant 响应添加到顶层
        result_item["assistant"] = response
        
        # [修改] 添加完整的、修改后的字典
        results.append(result_item)
        completed += 1
        # 写入进度计数（尽量低频，但保持简单每条写一次，文件极小）
        if progress_file and (completed % 1 == 0):
            try:
                with open(progress_file, 'w') as pf:
                    pf.write(str(completed) + "\n")
            except Exception:
                pass
        
    # 末尾标记完成
    if progress_file:
        try:
            with open(progress_file, 'w') as pf:
                pf.write("DONE\n")
        except Exception:
            pass
    return results

def main():
    """主执行函数"""
    args = parse_args()
    rank, world_size = setup_distributed()

    # 从命令行参数构建基础生成配置
    base_generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty
    }

    if rank == 0:
        print("="*60)
        print(f"🚀 开始分布式推理任务，共 {world_size} 个 GPU。")
        print(f"📂 Prompt 文件: {args.prompt_file_path} (Key: {args.prompt_key})")
        print(f"🤖 模型列表: {args.model_paths}")
        print(f"🔥 温度列表: {args.temperatures}")
        print(f"📝 输出目录: {args.output_dir}")
        print("="*60)
        os.makedirs(args.output_dir, exist_ok=True)
        total_runs = len(args.model_paths) * len(args.temperatures)
        main_progress = tqdm(total=total_runs, desc="总进度", position=0)

    # 1. [修改] 所有 rank 都加载完整数据列表
    all_data = load_data(args.prompt_file_path, args.prompt_key)
    if not all_data:
        if rank == 0:
            print("❌ Prompt 文件为空或无法读取，程序退出。")
        cleanup_distributed()
        return
        
    # 主循环：遍历模型和温度
    for model_path in args.model_paths:
        torch.cuda.empty_cache()

        # 基于模型路径构建一个清晰、适合做文件名的 model_name
        slice_name = os.path.basename(model_path)
        parent_dir_name = os.path.basename(os.path.dirname(model_path))
        model_name = f"{parent_dir_name}_{slice_name}"

        if rank == 0:
            print(f"\n🔄 正在加载模型: {model_name}...")
        
        if world_size > 1:
            dist.barrier()
            
        for temp in args.temperatures:
            if rank == 0:
                main_progress.set_description(f"模型: {model_name}, Temp: {temp}")

            # 2. [修改] 每个 rank 获取自己的数据子集
            data_on_this_rank = all_data[rank::world_size]

            # 3. [修改] 在当前 rank 上执行推理，传入 prompt_key
            # 为该温度建立跨 rank 进度监控（仅 rank0）
            monitor_thread = None
            stop_monitor = threading.Event()
            if rank == 0 and world_size > 1:
                progress_dir = os.path.join(args.output_dir, f".progress_local_temp_{temp}")
                os.environ["AA_LOCAL_PROGRESS_DIR"] = progress_dir
                total = len(all_data)

                def monitor_func():
                    bar = tqdm(total=total, desc="样本总进度", position=2, leave=False)
                    last = 0
                    try:
                        while not stop_monitor.is_set():
                            done = 0
                            try:
                                for r in range(world_size):
                                    pfile = os.path.join(progress_dir, f"rank_{r}.progress")
                                    if os.path.exists(pfile):
                                        with open(pfile, 'r') as pf:
                                            content = pf.read().strip()
                                            if content == "DONE":
                                                # 该 rank 完成数量即为该 rank 分配的样本数
                                                done += len(all_data[r::world_size])
                                            else:
                                                try:
                                                    done += int(content)
                                                except Exception:
                                                    pass
                            except Exception:
                                pass
                            inc = max(done - last, 0)
                            if inc:
                                bar.update(inc)
                                last = done
                            # 简单尾延迟观测：计算每个 rank 未完成数量
                            inflight = total - done
                            bar.set_postfix({"inflight": inflight})
                            time.sleep(1.0)
                    finally:
                        bar.close()

                monitor_thread = threading.Thread(target=monitor_func, daemon=True)
                monitor_thread.start()
            local_results = run_inference_on_rank(
                rank, 
                world_size, 
                model_path, 
                temp, 
                data_on_this_rank, 
                base_generation_config,
                args.prompt_key # [修改] 传入 key
            )

            # 4. 收集所有 rank 的结果到 rank 0
            all_results_gathered = None
            if world_size > 1:
                all_results_gathered = [None] * world_size
                dist.all_gather_object(all_results_gathered, local_results)
            else:
                all_results_gathered = [local_results]
            
            # 5. Rank 0 负责写入文件
            if rank == 0:
                # [修改] 重新排序，以匹配原始数据顺序
                final_results = [None] * len(all_data)
                for i in range(world_size):
                    final_results[i::world_size] = all_results_gathered[i]
                
                # 定义输出文件名
                output_filename = f"temp-{temp}.jsonl"
                output_path = os.path.join(args.output_dir, output_filename)
                
                num_written = 0
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in final_results:
                        if item: # 确保 item 不是 None
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                            num_written += 1
                
                print(f"\n✅ 结果已保存至: {output_path}")
                main_progress.update(1)

                # 写出每温度指标（仅 rank 0 记录）
                write_metric(
                    args.metric_file,
                    step="batch_inference_local",
                    data={
                        "model": model_name,
                        "temperature": temp,
                        "output_path": output_path,
                        "num_prompts": len(all_data),
                        "num_outputs": num_written,
                    }
                )

            if world_size > 1:
                dist.barrier()

            # 停止与回收监控
            if rank == 0 and monitor_thread is not None:
                stop_monitor.set()
                monitor_thread.join(timeout=2)

    if rank == 0:
        main_progress.close()
        print("\n🎉 所有推理任务完成！")
        write_metric(
            args.metric_file,
            step="batch_inference_local_summary",
            data={
                "output_dir": args.output_dir,
                "temperatures": args.temperatures,
                "models": args.model_paths,
            }
        )

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
