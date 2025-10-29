#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多供应商 API 批量推理脚本 (接口统一版)

功能:
- 接口与 `batch_inference_local.py` 兼容。
- 新增 `--provider` 参数，用于选择 API 供应商。
- 支持: 'openai', 'anthropic', 'deepseek', 'google' (通过自定义 Base URL)。
- 从 `.env` 文件自动加载统一的 NEWAPI_API_KEY 和 各自的 Base URL。
- [V2.5 (并发版)] 使用 ThreadPoolExecutor 实现并发请求。
- [V2.5 (并发版)] 将关键参数移至顶部的 CONFIG 类，方便修改。
- [V2.5 (并发版)] 回退 V2.4 的修改，仅对 429 速率限制错误进行指数退避。
- [V2.6] 将 'gemini' 重命名为 'google'。
- 为每个温度生成一致的 JSONL 格式输出。
- 可扩展设计：使用抽象基类和工厂模式，方便未来添加新的 API。

依赖库 (请确保已安装):
    pip install "openai>=1.0" "anthropic>=0.20" "python-dotenv" "tqdm" "requests"

作者: 由 Gemini 为 align-anything 框架生成 (V2.6 并发版)
"""

import os
import json
import time
import argparse
import abc  # 导入抽象基类模块
import random # 用于指数退避的抖动
import concurrent.futures # 用于并发处理
import functools # 用于偏函数
from tqdm import tqdm
from metrics import write_metric
from dotenv import load_dotenv
import requests # 用于 Google REST API

# 导入各供应商 SDK
from openai import OpenAI, APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError
import importlib
try:
    anthropic = importlib.import_module('anthropic')
except Exception:
    anthropic = None

# ==============================================================================
# --- 脚本核心配置 ---
# ==============================================================================
class CONFIG:
    """
    在此处统一配置脚本的关键参数
    """
    
    # --- 并发与重试配置 ---
    
    # 最大并发请求数
    # 调高此值可加快处理速度，但请注意不要超过您 API 代理的速率限制
    MAX_CONCURRENT_REQUESTS = 10 
    
    # API 请求失败时的最大重试次数 (仅限 429 错误)
    MAX_RETRIES = 5

    # 指数退避的基础等待时间（秒）
    # 第一次重试等待 (10s * 2^0) +抖动, 第二次 (10s * 2^1) +抖动, ...
    BASE_WAIT_TIME_SECONDS = 1

    # --- Prompt 内容配置 ---

    # 系统提示语
    SYSTEM_PROMPT = "You are a helpful assistant."

    # 添加到用户 prompt 末尾的指令后缀
    USER_PROMPT_SUFFIX = '\nAttention: This is all the information. Please provide the most likely protein sequence. The answer should only contain the single-letter codes for the 20 amino acids that make up the protein. Do not use three-letter codes or include any other characters, including spaces.'
    USER_PROMPT_SUFFIX = ''
# ==============================================================================
# --- 配置结束 ---
# ==============================================================================


def parse_args():
    """
    解析命令行参数。
    接口与 `batch_inference_local.py` 保持一致, 新增 --provider。
    """
    parser = argparse.ArgumentParser(
        description="多供应商 API 批量推理脚本 (接口统一版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- 新增参数 ---
    parser.add_argument(
        "--provider", type=str, required=True,
        choices=['openai', 'anthropic', 'deepseek', 'google'], # [V2.6] gemini -> google
        help="选择要调用的 API 供应商。"
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
        help="重复惩罚系数 (仅部分 API 支持)。"
    )
    parser.add_argument(
        "--metric_file", type=str, default=None,
        help="结构化指标输出文件 (JSONL)，可选。"
    )
    
    return parser.parse_args()

def load_prompts(file_path, prompt_key):
    """从 JSONL 文件加载所有 prompts (与 local 脚本一致)"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prompts.append(json.loads(line)[prompt_key])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"警告: 跳过格式错误或缺少键 '{prompt_key}' 的行: {line.strip()} - 错误: {e}")
    except FileNotFoundError:
        print(f"错误: Prompt 文件未找到 -> {file_path}")
        return None
    return prompts

# --- API 客户端抽象基类 ---
class AbstractChatClient(abc.ABC):
    """
    API 客户端的抽象基类，定义了统一的接口。
    """
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = args # 存储所有命令行参数以备后用
        if args.repetition_penalty != 1.0:
            self.warn_repetition_penalty()

    def warn_repetition_penalty(self):
        # 默认警告，子类可以覆盖
        print(f"警告: {self.__class__.__name__} 可能不完全支持 repetition_penalty，该参数将被近似或忽略。")

    @abc.abstractmethod
    def generate(self, prompt, temperature):
        """
        所有子类必须实现的推理方法。
        必须返回一个字符串 (生成的文本)。
        """
        pass
    
    def is_rate_limit_error(self, error):
        """
        [V2.5 回退] 检查传入的异常是否为 429 速率限制错误。
        """
        return False

# --- OpenAI 和 DeepSeek (兼容) 客户端实现 ---
class OpenAICompatibleClient(AbstractChatClient):
    """
    此类同时适用于 OpenAI 和 DeepSeek，因为它们共享相同的 API 格式。
    """
    def __init__(self, model_name, args, api_key, base_url):
        super().__init__(model_name, args)
        
        # 自动修正 .env 文件中多余的 /chat/completions 后缀
        suffix_to_remove = "/chat/completions"
        if base_url.endswith(suffix_to_remove):
            original_url = base_url
            base_url = base_url[:-len(suffix_to_remove)]
            print(f"警告: 检测到 base_url 包含 '{suffix_to_remove}'。")
            print(f"     已自动将其修正为: {base_url} (原始: {original_url})")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"OpenAI 兼容客户端初始化完成，模型: {model_name}，URL: {base_url}")

    def warn_repetition_penalty(self):
        print(f"信息: 'repetition_penalty' (值: {self.args.repetition_penalty}) 将被映射到 OpenAI 的 'frequency_penalty'。")

    def generate(self, prompt, temperature):
        messages = [
            {"role": "system", "content": CONFIG.SYSTEM_PROMPT},
            {"role": "user", "content": prompt + CONFIG.USER_PROMPT_SUFFIX}
        ]
        # 映射 repetition_penalty -> frequency_penalty
        freq_penalty = 0.0 if self.args.repetition_penalty == 1.0 else (self.args.repetition_penalty - 1.0)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature > 0 else 0.0,
            max_tokens=self.args.max_new_tokens,
            # [V2.2 修复] 移除了 top_p 参数
            frequency_penalty=freq_penalty
        )
        return response.choices[0].message.content.strip()

    def is_rate_limit_error(self, error):
        """[V2.5 回退] 检查 OpenAI SDK 抛出的是否为 429 错误"""
        return isinstance(error, OpenAIRateLimitError) or \
               (isinstance(error, OpenAIAPIError) and error.status_code == 429)


# --- Anthropic (Claude) 客户端实现 ---
class AnthropicClient(AbstractChatClient):
    def __init__(self, model_name, args, api_key, base_url):
        super().__init__(model_name, args)
        
        suffix_to_remove = "/messages"
        if base_url.endswith(suffix_to_remove):
            original_url = base_url
            base_url = base_url[:-len(suffix_to_remove)]
            print(f"警告: 检测到 base_url 包含 '{suffix_to_remove}'。")
            print(f"     已自动将其修正为: {base_url} (原始: {original_url})")
        
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        print(f"Anthropic 客户端初始化完成，模型: {model_name}，URL: {base_url}")

    def warn_repetition_penalty(self):
        print("警告: Anthropic (Claude) API 不支持 repetition_penalty，该参数将被忽略。")

    def generate(self, prompt, temperature):
        response = self.client.messages.create(
            model=self.model_name,
            system=CONFIG.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt + CONFIG.USER_PROMPT_SUFFIX}
            ],
            temperature=temperature,
            max_tokens=self.args.max_new_tokens,
            top_p=self.args.top_p
        )
        return response.content[0].text.strip()

    def is_rate_limit_error(self, error):
        """[V2.5 回退] 检查 Anthropic SDK 抛出的是否为 429 错误"""
        return isinstance(error, anthropic.RateLimitError) or \
               (isinstance(error, anthropic.APIError) and error.status_code == 429)


# --- [V2.6] Google (REST API) 客户端实现 ---
class GoogleRestClient(AbstractChatClient):
    """
    此类使用 'requests' 库来精确模拟您的 curl 示例。
    [V2.6] 重命名自 GeminiRestClient
    """
    def __init__(self, model_name, args, api_key, base_url):
        super().__init__(model_name, args)
        self.api_key = api_key
        self.url = f"{base_url}/models/{model_name}:generateContent?key={self.api_key}"
        print(f"Google (REST) 客户端初始化完成，模型: {model_name}，URL: {base_url}/models/{model_name}:...")

    def warn_repetition_penalty(self):
        print("警告: Google (Gemini) API 不支持 repetition_penalty，该参数将被忽略。")

    def generate(self, prompt, temperature):
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt + CONFIG.USER_PROMPT_SUFFIX}]}
            ],
            "systemInstruction": {
                "parts": [{"text": CONFIG.SYSTEM_PROMPT}]
            },
            "generationConfig": {
                "temperature": temperature,
                "topP": self.args.top_p,
                "maxOutputTokens": self.args.max_new_tokens,
                "candidateCount": 1
            }
        }
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status() 
        data = response.json()
        
        if "candidates" not in data or not data["candidates"]:
            print(f"警告: Google API 返回了空 'candidates'。响应: {data}")
            return "GOOGLE_ERROR: No candidates returned."
            
        return data['candidates'][0]['content']['parts'][0]['text'].strip()

    def is_rate_limit_error(self, error):
        """[V2.5 回退] 检查 requests 抛出的是否为 429 错误"""
        return isinstance(error, requests.HTTPError) and error.response.status_code == 429

# --- 客户端工厂函数 ---
def get_client(provider, model_name, args):
    """
    根据 provider 名称和模型名称，初始化并返回对应的 API 客户端。
    """
    api_key = os.getenv("NEWAPI_API_KEY")
    if not api_key:
        raise ValueError("错误: 未找到 NEWAPI_API_KEY，请检查 .env 文件。")

    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("错误: OPENAI_BASE_URL 未在 .env 文件中设置。")
        return OpenAICompatibleClient(model_name, args, api_key, base_url)
        
    elif provider == "deepseek":
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not base_url:
            raise ValueError("错误: DEEPSEEK_BASE_URL 未在 .env 文件中设置。")
        return OpenAICompatibleClient(model_name, args, api_key, base_url)

    elif provider == "anthropic":
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if not base_url:
            raise ValueError("错误: ANTHROPIC_BASE_URL 未在 .env 文件中设置。")
        return AnthropicClient(model_name, args, api_key, base_url)
        
    elif provider == "google": # [V2.6] gemini -> google
        base_url = os.getenv("GOOGLE_BASE_URL") # [V2.6] GEMINI_BASE_URL -> GOOGLE_BASE_URL
        if not base_url:
            raise ValueError("错误: GOOGLE_BASE_URL 未在 .env 文件中设置。")
        return GoogleRestClient(model_name, args, api_key, base_url) # [V2.6] GeminiRestClient -> GoogleRestClient
        
    else:
        raise ValueError(f"错误: 不支持的 provider: {provider}")

# --- [V2.5 新增] 单个 prompt 的处理函数 (用于并发) ---
def process_single_prompt(prompt_text, client, temp):
    """
    处理单个 prompt，包含完整的指数退避重试逻辑。
    此函数被设计为在并发线程中运行。
    
    返回: 一个字典, {"prompt": "...", "assistant": "..."}
    """
    generated_text = ""
    
    for attempt in range(CONFIG.MAX_RETRIES):
        try:
            # 1. 尝试调用 API
            generated_text = client.generate(
                prompt=prompt_text,
                temperature=temp
            )
            # 2. 如果成功，跳出重试循环
            break 
            
        except (OpenAIAPIError, anthropic.APIError, requests.RequestException) as e:
            
            # 3. 检查是否为 429 速率限制错误
            if client.is_rate_limit_error(e):
                if attempt < CONFIG.MAX_RETRIES - 1:
                    # 计算退避时间：(2^n * 基础时间) + 随机抖动
                    wait_time = (CONFIG.BASE_WAIT_TIME_SECONDS * (2 ** attempt)) + random.uniform(0, 3)
                    print(f"\n警告: 收到 429 速率限制错误 (Prompt: '{prompt_text[:20]}...'). "
                          f"将在 {wait_time:.1f} 秒后重试 (第 {attempt + 1}/{CONFIG.MAX_RETRIES} 次)...")
                    time.sleep(wait_time)
                else:
                    # 4. 达到最大重试次数
                    print(f"\n错误: 达到最大重试次数 ({CONFIG.MAX_RETRIES})。放弃此 prompt。错误: {e}")
                    generated_text = f"API_ERROR: Max retries exceeded for Rate Limit. {e}"
                    break # 放弃并跳出循环
            else:
                # 5. 如果是其他 API 错误 (如 400, 500, 503)
                print(f"\n错误: API 调用失败 (不可重试错误): {e}")
                generated_text = f"API_ERROR: {e}"
                break # 不重试非 429 错误
                
        except Exception as e:
            # 6. 捕获其他未知 Python 错误
            print(f"\n错误: 发生未知错误 (Prompt: '{prompt_text[:50]}...'): {e}")
            generated_text = f"UNKNOWN_ERROR: {e}"
            break # 不重试未知错误

    # 7. 返回结果
    return {
        "prompt": prompt_text,
        "assistant": generated_text 
    }

# --- 主执行函数 (V2.5 更新) ---
def main():
    """主执行函数"""
    args = parse_args()
    
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    model_name = args.model_paths[0]
    
    print("="*60)
    print(f"🚀 开始 API 推理任务 (并发版 v2.6)...") # [V2.6]
    print(f"⚡ 最大并发数: {CONFIG.MAX_CONCURRENT_REQUESTS}")
    print(f"🏢 供应商: {args.provider}")
    print(f"🤖 API 模型: {model_name}")
    print(f"📂 Prompt 文件: {args.prompt_file_path}")
    print(f"🔥 温度列表: {args.temperatures}")
    print(f"📝 输出目录: {args.output_dir}")
    print("="*60)
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_prompts = load_prompts(args.prompt_file_path, args.prompt_key)
    if not all_prompts:
        print("❌ Prompt 文件为空或无法读取，程序退出。")
        return

    try:
        # 初始化一个客户端
        client = get_client(args.provider, model_name, args)
    except ValueError as e:
        print(e)
        return

    for temp in args.temperatures:
        print(f"\n🔄 正在处理温度: {temp}...")
        
        output_filename = f"temp-{temp}.jsonl"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # [V2.5 更新] 使用 ThreadPoolExecutor 进行并发处理
        num_written = 0
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # 创建一个偏函数，固定 client 和 temp 参数
            worker_func = functools.partial(process_single_prompt, client=client, temp=temp)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG.MAX_CONCURRENT_REQUESTS) as executor:
                # executor.map 会保持与 all_prompts 相同的顺序返回结果
                results_iterator = executor.map(worker_func, all_prompts)
                
                # 使用 tqdm 包装迭代器以显示进度
                for output_record in tqdm(results_iterator, total=len(all_prompts), desc=f"Temp-{temp} Infer"):
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    num_written += 1
        
        print(f"✅ 结果已保存至: {output_path}")
        write_metric(
            args.metric_file,
            step="batch_inference_api_multi",
            data={
                "provider": args.provider,
                "model": model_name,
                "temperature": temp,
                "output_path": output_path,
                "num_prompts": len(all_prompts),
                "num_outputs": num_written,
            }
        )
        
    print("\n🎉 所有 API 推理任务完成！")

if __name__ == "__main__":
    main()

