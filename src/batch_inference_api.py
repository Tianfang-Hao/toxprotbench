#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 API 批量推理脚本 (V3.3)

功能:
- 接口与 `batch_inference_local.py` 兼容 (移除了 --provider)。
- 假设所有 API 均与 OpenAI 格式兼容。
- 从 `.env` 文件自动加载统一的 NEWAPI_API_KEY 和 API_BASE_URL。
- [V2.5 (并发版)] 使用 ThreadPoolExecutor 实现并发请求。
- [V2.5 (并发版)] 将关键参数移至顶部的 CONFIG 类，方便修改。
- [V2.5 (并发版)] 仅对 429 速率限制错误进行指数退避。
- [V2.7] 为所有 API 客户端添加统一的请求超时 (REQUEST_TIMEOUT_SECONDS)。
- [V3.1] 移除 .strip() 以保留原始输出。
- [V3.2] 在输出的 JSONL 中额外添加 'finish_reason' 字段。
- [V3.3] 输出的 JSONL 保留输入 JSONL 的所有字段，并将 'assistant' 
          和 'finish_reason' 作为新字段并列添加。

依赖库 (请确保已安装):
    pip install "openai>=1.0" "python-dotenv" "tqdm"

作者: 由 Gemini 为 align-anything 框架生成 (V3.3 统一版)
"""

import os
import json
import time
import argparse
import random # 用于指数退避的抖动
import concurrent.futures # 用于并发处理
import functools # 用于偏函数
from tqdm import tqdm
from metrics import write_metric, metric_timer
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple

# 导入各供应商 SDK
from openai import (
    OpenAI,
    APIError as OpenAIAPIError,
    RateLimitError as OpenAIRateLimitError,
    APITimeoutError as OpenAIAPITimeoutError,
)

# ==============================================================================
# --- 脚本核心配置 ---
# ==============================================================================
class CONFIG:
    """
    在此处统一配置脚本的关键参数
    """
    
    # --- 并发、重试与超时配置 ---
    # 最大并发请求数
    MAX_CONCURRENT_REQUESTS = 10 
    
    # API 请求失败时的最大重试次数 (仅限 429 错误)
    MAX_RETRIES = 5 

    # API 超时的最大重试次数
    MAX_TIMEOUT_RETRIES = 3

    # 任务在遇到“硬超时(硬性总期限)”或返回超时占位后，最高允许的重新投递次数
    # 注：这是在 process_single_prompt 内部超时重试之外的“任务级”重试次数
    MAX_RESPAWN_ON_TIMEOUT = 2

    # 指数退避的基础等待时间（秒）
    BASE_WAIT_TIME_SECONDS = 1

    # API 请求的全局超时时间（秒）
    REQUEST_TIMEOUT_SECONDS = 1500
    # 每个样本的硬性总期限（秒）：用于兜底防卡死，超过即标记为超时写占位并跳过等待
    HARD_DEADLINE_MARGIN_SECONDS = 2000
    # 将任务标记为“慢任务”的阈值（秒），用于实时可视化尾部延迟
    SLOW_TASK_SLA_SECONDS = 20

    # --- Prompt 内容配置 ---

    # 系统提示语
    SYSTEM_PROMPT = "You are a helpful assistant."

    # 添加到用户 prompt 末尾的指令后缀
    # USER_PROMPT_SUFFIX = ''
    USER_PROMPT_SUFFIX = '\nAttention: This is all the information. Please provide the most likely protein sequence. The answer should only contain a single sequence of single-letter codes for the 20 amino acids that make up the protein. Do not use three-letter codes. Do not include any.'
# ==============================================================================
# --- 配置结束 ---
# ==============================================================================


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="统一 API 批量推理脚本 (V3.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 文件 I/O 配置 (与 local 脚本一致) ---
    parser.add_argument(
        "--prompt_file_path", type=str, required=True,
        help="输入的 prompt 文件路径 (JSONL 格式)。"
    )
    parser.add_argument(
        "--prompt_key", type=str, default="generated_prompt_for_sequence_model",
        help="JSONL 文件中包含 prompt 文本的键 (key)。"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出结果的根目录。"
    )

    # --- 模型与推理配置 (与 local 脚本一致) ---
    parser.add_argument(
        "--model_paths", type=str, nargs='+', required=True,
        help="需要进行推理的一个模型名称 (例如 'gpt-5' 或 'claude-sonnet-4-5-20250929')。"
    )
    parser.add_argument(
        "--temperatures", type=float, nargs='+', required=True,
        help="需要测试的一个或多个温度值。"
    )

    # --- 生成参数配置 (与 local 脚本一致) ---
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="生成的最大 token 数量。"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8,
        help="Top-p (nucleus) 采样参数。"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="重复惩罚系数 (映射到 frequency_penalty)。"
    )
    parser.add_argument(
        "--metric_file", type=str, default=None,
        help="结构化指标输出文件 (JSONL)，可选。"
    )
    # --- 历史结果复用（可选） ---
    parser.add_argument(
        "--reuse_dir", type=str, default=None,
        help="可选：指定一个历史输出目录，若该目录下存在 temp-{T}.jsonl，则对已存在的样本直接复用结果；若该样本的 finish_reason 为 'api_error' 则不复用而重新推理。"
    )
    parser.add_argument(
        "--reuse_key", type=str, default="id",
        help="用于匹配样本的键，默认使用 'id'；若缺失则退化为使用 prompt_key 对应的文本作为匹配键。"
    )
    parser.add_argument(
        "--reuse_dirs", type=str, nargs='+', default=None,
        help="可选：提供多个历史输出目录，将递归扫描所有目录下的 temp-{T}.jsonl 并合并复用映射（优先于 --reuse_dir）。"
    )
    
    return parser.parse_args()

def load_prompt_data(file_path, prompt_key):
    """
    [V3.3] 从 JSONL 文件加载所有 prompts，保留原始字典结构。
    """
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if prompt_key not in record:
                        print(f"警告: 跳过缺少键 '{prompt_key}' 的行: {line.strip()}")
                        continue
                    data_list.append(record)
                except (json.JSONDecodeError) as e:
                    print(f"警告: 跳过格式错误的行: {line.strip()} - 错误: {e}")
    except FileNotFoundError:
        print(f"错误: Prompt 文件未找到 -> {file_path}")
        return None
    return data_list

# --- [V3.0] 统一的 API 客户端 ---
class ApiClient:
    """
    此类使用 OpenAI SDK，用于所有兼容 OpenAI 格式的 API 调用。
    """
    def __init__(self, model_name, args, api_key, base_url):
        self.model_name = model_name
        self.args = args # 存储所有命令行参数以备后用
        if args.repetition_penalty != 1.0:
            print(f"信息: 'repetition_penalty' (值: {self.args.repetition_penalty}) 将被映射到 OpenAI 的 'frequency_penalty'。")

        # 自动修正 .env 文件中多余的 /chat/completions 后缀
        suffix_to_remove = "/chat/completions"
        if base_url.endswith(suffix_to_remove):
            original_url = base_url
            base_url = base_url[:-len(suffix_to_remove)]
            print(f"警告: 检测到 base_url 包含 '{suffix_to_remove}'。")
            print(f"     已自动将其修正为: {base_url} (原始: {original_url})")

        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=CONFIG.REQUEST_TIMEOUT_SECONDS 
        )
        print(f"统一 API 客户端初始化完成，模型: {model_name}，URL: {base_url}")


    def generate(self, prompt, temperature):
        messages = [
            {"role": "system", "content": CONFIG.SYSTEM_PROMPT},
            {"role": "user", "content": prompt + CONFIG.USER_PROMPT_SUFFIX}
        ]
        # 映射 repetition_penalty -> frequency_penalty
        freq_penalty = 0.0 if self.args.repetition_penalty == 1.0 else (self.args.repetition_penalty - 1.0)
        
        # [V2.2 修复] 移除了 top_p 参数
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature > 0 else 0.0,
            max_tokens=self.args.max_new_tokens,
            frequency_penalty=freq_penalty
        )
        
        # [V3.2] 获取内容和结束原因
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # [V3.2] 返回内容和结束原因 (V3.1 移除了 .strip())
        return content, finish_reason

    def is_rate_limit_error(self, error):
        """[V2.5 回退] 检查 OpenAI SDK 抛出的是否为 429 错误"""
        if isinstance(error, OpenAIRateLimitError):
            return True
        # 某些异常类型可能没有 status_code 属性
        if isinstance(error, OpenAIAPIError):
            code = getattr(error, 'status_code', None)
            return code == 429
        return False


# --- [V2.5 新增] 单个 prompt 的处理函数 (用于并发) ---
def process_single_prompt(input_data, client, temp, prompt_key):
    """
    [V3.3] 处理单个 prompt 数据 (字典)，包含完整的指数退避重试逻辑。
    此函数被设计为在并发线程中运行。
    
    [V3.3] 返回: 一个字典, 包含 input_data 的所有字段，并添加 "assistant" 和 "finish_reason"。
    """
    
    # [V3.3] 复制原始数据以保留所有字段
    output_record = input_data.copy()
    generated_text = ""
    finish_reason = "unknown" 

    try:
        prompt_text = input_data[prompt_key]
    except KeyError:
        print(f"\n错误: 在 process_single_prompt 中未找到 prompt_key '{prompt_key}'。数据: {input_data}")
        output_record['assistant'] = "ERROR: Missing prompt_key in input data."
        output_record['finish_reason'] = "preprocessing_error"
        return output_record
    
    rate_attempts = 0
    timeout_attempts = 0
    while True:
        try:
            # 1. 尝试调用 API
            generated_text, finish_reason = client.generate(
                prompt=prompt_text,
                temperature=temp
            )
            # 成功
            break

        except OpenAIAPITimeoutError as e:
            # 2. 超时重试（与 429 分开计数）
            if timeout_attempts < CONFIG.MAX_TIMEOUT_RETRIES:
                wait_time = (CONFIG.BASE_WAIT_TIME_SECONDS * (2 ** timeout_attempts)) + random.uniform(0, 3)
                timeout_attempts += 1
                print(f"\n警告: API 超时，将在 {wait_time:.1f}s 后重试 (timeout {timeout_attempts}/{CONFIG.MAX_TIMEOUT_RETRIES}).")
                time.sleep(wait_time)
                continue
            else:
                print(f"\n错误: API 超时重试已达上限 ({CONFIG.MAX_TIMEOUT_RETRIES})，放弃该样本。")
                generated_text = f"API_TIMEOUT: {e}"
                finish_reason = "timeout"
                break

        except OpenAIRateLimitError as e:
            # 3. 速率限制（429）重试
            if rate_attempts < CONFIG.MAX_RETRIES:
                wait_time = (CONFIG.BASE_WAIT_TIME_SECONDS * (2 ** rate_attempts)) + random.uniform(0, 3)
                rate_attempts += 1
                print(f"\n警告: 429 速率限制，将在 {wait_time:.1f}s 后重试 (rate {rate_attempts}/{CONFIG.MAX_RETRIES}).")
                time.sleep(wait_time)
                continue
            else:
                print(f"\n错误: 429 重试已达上限 ({CONFIG.MAX_RETRIES})，放弃该样本。")
                generated_text = f"API_ERROR: Max retries exceeded for Rate Limit. {e}"
                finish_reason = "max_retries_exceeded"
                break

        except OpenAIAPIError as e:
            # 4. 其他 API 错误（400/500等），不重试
            print(f"\n错误: API 调用失败 (status_code={getattr(e,'status_code',None)}): {e}")
            generated_text = f"API_ERROR: {e}"
            finish_reason = "api_error"
            break

        except Exception as e:
            # 5. 其他未知错误，不重试
            print(f"\n错误: 发生未知错误 (Prompt: '{prompt_text[:50]}...'): {e}")
            generated_text = f"UNKNOWN_ERROR: {e}"
            finish_reason = "unknown_error"
            break

    # 7. [V3.3] 将结果合并回原始字典
    output_record['assistant'] = generated_text
    output_record['finish_reason'] = finish_reason
    return output_record


def build_sample_key(record: Dict[str, Any], prompt_key: str, reuse_key: str) -> str:
    """生成用于匹配的样本键：优先使用 reuse_key；若不存在则用 prompt 文本。"""
    if reuse_key and reuse_key in record:
        return str(record.get(reuse_key))
    return str(record.get(prompt_key, ""))


def _iter_temp_files(reuse_dirs: List[str], temp: float) -> List[str]:
    target_name = f"temp-{temp}.jsonl"
    files: List[str] = []
    for d in reuse_dirs:
        if not d:
            continue
        if os.path.isfile(d) and os.path.basename(d) == target_name:
            files.append(d)
            continue
        if not os.path.isdir(d):
            continue
        for root, _, fls in os.walk(d):
            for fn in fls:
                if fn == target_name:
                    files.append(os.path.join(root, fn))
    return files


def _infer_model_and_domain_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    """从文件路径推断 (model, domain)。
    规则：路径片段中若包含 'Animal' 或 'Bacteria'，则将其作为 domain，且其前一个片段作为 model。
    例：.../results/<run>/<model>/Bacteria/raw/temp-0.7.jsonl → (model, 'Bacteria')
    """
    parts = os.path.normpath(path).split(os.sep)
    domain = None
    model = None
    for i, p in enumerate(parts):
        if p in ("Animal", "Bacteria"):
            domain = p
            if i > 0:
                model = parts[i - 1]
            break
    return model, domain


def _infer_domain_from_output_dir(output_dir: str) -> Optional[str]:
    parts = os.path.normpath(output_dir).split(os.sep)
    if "Animal" in parts:
        return "Animal"
    if "Bacteria" in parts:
        return "Bacteria"
    return None


def load_reuse_map(
    reuse_dirs: List[str],
    temp: float,
    prompt_key: str,
    reuse_key: str,
    current_model: str,
    current_domain: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """从一个或多个复用目录（递归扫描）加载 temp-{temp}.jsonl，构建 key -> {assistant, finish_reason} 的映射。
    仅保留与当前 (model, domain) 一致的条目，避免跨模型/跨物种复用。
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    files = _iter_temp_files(reuse_dirs, temp)
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # 推断该记录的 model/domain
                    rec_model = rec.get('model')
                    rec_domain = rec.get('domain')
                    if not rec_model or not rec_domain:
                        pm, pd = _infer_model_and_domain_from_path(path)
                        rec_model = rec_model or pm
                        rec_domain = rec_domain or pd

                    # 过滤：必须与当前 (model, domain) 匹配（若 current_domain 无法推断，则仅比对 model）
                    if rec_model != current_model:
                        continue
                    if current_domain is not None and rec_domain != current_domain:
                        continue

                    key = build_sample_key(rec, prompt_key, reuse_key)
                    if not key:
                        continue
                    mapping[key] = {
                        'assistant': rec.get('assistant', ''),
                        'finish_reason': rec.get('finish_reason', ''),
                    }
        except Exception:
            continue
    return mapping


def is_error_like_output(finish_reason: str, assistant_text: str) -> bool:
    """判断历史结果是否属于错误/不可复用：所有错误都强制重推；只要是模型真实输出就可复用。
    规则：
    - finish_reason 在以下集合则视为错误：{api_error, timeout, timeout_hard, max_retries_exceeded, worker_error, unknown_error}
    - 或 assistant 文本以已知错误前缀开头：API_TIMEOUT, API_ERROR, UNKNOWN_ERROR, WORKER_ERROR
    - 其他（如 stop/length/content_filter/正常文本）视为可复用。
    """
    fr = (finish_reason or "").lower()
    if fr in {"api_error", "timeout", "timeout_hard", "max_retries_exceeded", "worker_error", "unknown_error"}:
        return True
    at = assistant_text or ""
    err_prefixes = ("API_TIMEOUT", "API_ERROR", "UNKNOWN_ERROR", "WORKER_ERROR")
    if any(at.startswith(p) for p in err_prefixes):
        return True
    return False

# --- 主执行函数 (V3.3 更新) ---
def main():
    """主执行函数"""
    args = parse_args()
    
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    model_name = args.model_paths[0]
    
    print("="*60)
    print(f"🚀 开始 统一API 推理任务 (并发版 v3.3)...") # [V3.3]
    print(f"⚡ 最大并发数: {CONFIG.MAX_CONCURRENT_REQUESTS}")
    print(f" timeout={CONFIG.REQUEST_TIMEOUT_SECONDS}s")
    print(f"🤖 API 模型: {model_name}")
    print(f"📂 Prompt 文件: {args.prompt_file_path}")
    print(f"🔥 温度列表: {args.temperatures}")
    print(f"📝 输出目录: {args.output_dir}")
    print("="*60)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # [V3.3] 加载完整的 prompt 数据，而不仅仅是文本
    all_prompt_data = load_prompt_data(args.prompt_file_path, args.prompt_key)
    if not all_prompt_data:
        print("❌ Prompt 文件为空或无法读取，程序退出。")
        return

    try:
        # [V3.0] 初始化统一客户端
        api_key = os.getenv("NEWAPI_API_KEY")
        if not api_key:
            raise ValueError("错误: 未找到 NEWAPI_API_KEY，请检查 .env 文件。")
            
        base_url = os.getenv("API_BASE_URL") # [V3.0] 使用新的统一 URL
        if not base_url:
            raise ValueError("错误: API_BASE_URL 未在 .env 文件中设置。")

        client = ApiClient(model_name, args, api_key, base_url)

    except ValueError as e:
        print(e)
        return

    for temp in args.temperatures:
        print(f"\n🔄 -----------------------------------")
        print(f"🔄 正在处理温度: {temp}...")
        print(f"🔄 -----------------------------------")
        
        output_filename = f"temp-{temp}.jsonl"
        output_path = os.path.join(args.output_dir, output_filename)
        
        num_written = 0
        num_reused = 0
        num_reinfer_due_to_error = 0
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # [V3.4] 使用“有界并发 + 结果兜底”的方式，避免单样本卡死导致进度条停在最后一个
            worker_func = functools.partial(
                process_single_prompt,
                client=client,
                temp=temp,
                prompt_key=args.prompt_key
            )

            hard_deadline_per_task = CONFIG.REQUEST_TIMEOUT_SECONDS * (CONFIG.MAX_TIMEOUT_RETRIES + 1) + CONFIG.HARD_DEADLINE_MARGIN_SECONDS

            # 记录每个样本的“任务级重试”尝试次数与是否曾发生超时
            attempts_by_idx = {}
            had_timeout_before = {}
            respawned_timeouts = 0
            recovered_timeouts = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG.MAX_CONCURRENT_REQUESTS) as executor:
                # 提交所有任务，记录起始时间与索引，便于兜底输出
                futures = {}
                start_times = {}
                # 历史复用映射（支持多个目录；若仅提供 --reuse_dir 也兼容）
                reuse_dirs: List[str] = []
                if args.reuse_dirs:
                    reuse_dirs.extend(args.reuse_dirs)
                elif args.reuse_dir:
                    reuse_dirs.append(args.reuse_dir)
                # 当前任务上下文
                current_domain = _infer_domain_from_output_dir(args.output_dir)
                reuse_map = load_reuse_map(
                    reuse_dirs, temp, args.prompt_key, args.reuse_key, model_name, current_domain
                ) if reuse_dirs else {}
                pbar = tqdm(total=len(all_prompt_data), desc=f"Temp-{temp} Infer")
                forced_timeouts = 0
                for idx, item in enumerate(all_prompt_data):
                    # 判断是否可复用
                    reused_here = False
                    if reuse_map:
                        key = build_sample_key(item, args.prompt_key, args.reuse_key)
                        if key in reuse_map:
                            prev = reuse_map[key]
                            if is_error_like_output(prev.get('finish_reason', ''), prev.get('assistant', '')):
                                # 标记需重推
                                num_reinfer_due_to_error += 1
                            else:
                                # 直接复用：以当前输入字段为基，覆盖 assistant/finish_reason
                                out = item.copy()
                                out['assistant'] = prev.get('assistant', '')
                                out['finish_reason'] = prev.get('finish_reason', '')
                                f_out.write(json.dumps(out, ensure_ascii=False) + '\n')
                                num_written += 1
                                num_reused += 1
                                pbar.update(1)
                                reused_here = True
                    if reused_here:
                        continue

                    fut = executor.submit(worker_func, item)
                    futures[fut] = idx
                    start_times[fut] = time.time()
                    attempts_by_idx[idx] = attempts_by_idx.get(idx, 0) + 1
                    had_timeout_before[idx] = False

                # 循环等待完成或超时的任务；定期检查卡住的 future 并兜底
                while futures:
                    done, not_done = concurrent.futures.wait(
                        futures.keys(), timeout=2, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    # 处理已完成的任务
                    for fut in done:
                        idx = futures.pop(fut)
                        start_times.pop(fut, None)
                        try:
                            output_record = fut.result()
                        except Exception as e:
                            # 不应常见；兜底写入错误占位
                            base = all_prompt_data[idx].copy()
                            base['assistant'] = f"WORKER_ERROR: {e}"
                            base['finish_reason'] = 'worker_error'
                            output_record = base

                        # 如果结果是“超时”类，占位或软超时，尝试任务级重投递
                        is_timeout_like = False
                        try:
                            fr = str(output_record.get('finish_reason', '')).lower()
                            assistant_txt = str(output_record.get('assistant', ''))
                            if fr in ('timeout', 'timeout_hard') or assistant_txt.startswith('API_TIMEOUT'):
                                is_timeout_like = True
                        except Exception:
                            pass

                        if is_timeout_like and attempts_by_idx.get(idx, 0) < CONFIG.MAX_RESPAWN_ON_TIMEOUT + 1:
                            # 记录曾发生超时
                            had_timeout_before[idx] = True
                            # 任务级重投递
                            respawned_timeouts += 1
                            fut2 = executor.submit(worker_func, all_prompt_data[idx])
                            futures[fut2] = idx
                            start_times[fut2] = time.time()
                            attempts_by_idx[idx] = attempts_by_idx.get(idx, 0) + 1
                            # 不写入占位，也不更新进度条，等待新的结果
                        else:
                            # 最终落盘
                            if had_timeout_before.get(idx, False) and not is_timeout_like:
                                recovered_timeouts += 1
                            f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                            num_written += 1
                            # 动态后缀更新（完成一个后也刷新一次）
                            now2 = time.time()
                            inflight = len(not_done)
                            slow = sum(1 for f in not_done if (now2 - start_times.get(f, now2)) > CONFIG.SLOW_TASK_SLA_SECONDS)
                            oldest_age = 0 if inflight == 0 else max((now2 - start_times.get(f, now2)) for f in not_done)
                            pbar.update(1)
                            pbar.set_postfix({
                                'inflight': inflight,
                                'slow': slow,
                                'oldest_s': f"{int(oldest_age)}",
                                'forced_to': forced_timeouts,
                            })

                    # 检查“硬性总期限”是否超时的未完成任务，做兜底输出并不再等待
                    now = time.time()
                    expired = [f for f in not_done if (now - start_times.get(f, now)) > hard_deadline_per_task]
                    for f in expired:
                        idx = futures.pop(f)
                        start_times.pop(f, None)
                        # 如果尚可进行任务级重投递，则重投而不是立刻写入占位
                        if attempts_by_idx.get(idx, 0) < CONFIG.MAX_RESPAWN_ON_TIMEOUT + 1:
                            had_timeout_before[idx] = True
                            respawned_timeouts += 1
                            fut2 = executor.submit(worker_func, all_prompt_data[idx])
                            futures[fut2] = idx
                            start_times[fut2] = time.time()
                            attempts_by_idx[idx] = attempts_by_idx.get(idx, 0) + 1
                            # 不更新 pbar，等待新的结果
                        else:
                            # 达到重投上限，落盘占位并计为强制超时
                            base = all_prompt_data[idx].copy()
                            base['assistant'] = "API_TIMEOUT: hard-deadline exceeded"
                            base['finish_reason'] = 'timeout_hard'
                            f_out.write(json.dumps(base, ensure_ascii=False) + '\n')
                            num_written += 1
                            forced_timeouts += 1
                            pbar.update(1)
                            # 刷新后缀
                            inflight2 = len(not_done) - 1 if len(not_done) > 0 else 0
                            slow2 = sum(1 for nf in not_done if (now - start_times.get(nf, now)) > CONFIG.SLOW_TASK_SLA_SECONDS)
                            oldest_age2 = 0 if inflight2 == 0 else max((now - start_times.get(nf, now)) for nf in not_done)
                            pbar.set_postfix({
                                'inflight': inflight2,
                                'slow': slow2,
                                'oldest_s': f"{int(oldest_age2)}",
                                'forced_to': forced_timeouts,
                            })

                pbar.close()
        
        print(f"✅ 结果已保存至: {output_path}")
        # 记录每个温度的一条指标
        write_metric(
            args.metric_file,
            step="batch_inference_api",
            data={
                "model": model_name,
                "temperature": temp,
                "output_path": output_path,
                "num_prompts": len(all_prompt_data),
                "num_outputs": num_written,
                "forced_timeouts": locals().get('forced_timeouts', 0),
                "respawned_timeouts": locals().get('respawned_timeouts', 0),
                "recovered_timeouts": locals().get('recovered_timeouts', 0),
            }
        )
        
    print("\n🎉 所有 API 推理任务完成！")
    # 记录任务级摘要
    write_metric(
        args.metric_file,
        step="batch_inference_api_summary",
        data={
            "model": model_name,
            "temperatures": args.temperatures,
            "output_dir": args.output_dir,
        }
    )

if __name__ == "__main__":
    main()