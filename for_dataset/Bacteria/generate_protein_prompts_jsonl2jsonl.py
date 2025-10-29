# -*- coding: utf-8 -*-
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiofiles
from tqdm.asyncio import tqdm as asyncio_tqdm

# 在程序开始时加载 .env 文件中的环境变量
# 这使得API密钥等敏感信息可以存储在.env文件中，而不是硬编码在代码里
load_dotenv()

# ==============================================================================
# --- 配置 (请根据您的需求修改) ---
# ==============================================================================
# 并发请求数量，根据您的API速率限制和机器性能调整
CONCURRENT_REQUESTS = 50
# 使用的语言模型名称
MODEL_NAME = "deepseek-chat"
# ★★★ 新的输入/输出文件名 ★★★
# 输入文件：包含源数据的JSONL文件
INPUT_FILE = 'VFDB_pure.jsonl'
# 源字段名：指定从输入JSONL文件的每个JSON对象中读取哪个字段的内容
# 例如，如果你的jsonl是 {"text": "这是蛋白质描述", "id": 1}, 那么这里应该设置为 "text"
SOURCE_FIELD_NAME = 'description'
# 成功处理后的输出文件
SUCCESS_OUTPUT_FILE = 'generated_prompts_from_jsonl.jsonl'
# 处理失败的记录输出文件
FAILURE_OUTPUT_FILE = 'jsonl_processing_failures.jsonl'
# API请求失败后的最大重试次数
MAX_RETRIES = 3
# 每次重试之间的延迟时间（秒）
RETRY_DELAY_SECONDS = 2
# ★★★ 您提供的Prompt模板 ★★★
# 这个模板中的 `{source_content}` 将会被替换为从JSONL文件指定字段中读取的实际内容
PROMPT_TEMPLATE = """
### Role ###
You are a top-tier Bioinformatics expert and Prompt Engineer. You excel at parsing brief database entries and inferring the complete biological picture of a protein.

### Task ###
Your mission is to convert a highly condensed protein description, extracted from a bacterial virulence factor database, into a detailed, precise, and technically-rich prompt. This generated prompt will be used to guide an advanced protein sequence generation model to create a functionally correct protein sequence.

### Chain of Thought & Execution Steps ###
To construct the highest quality prompt, you must internally follow this chain of thought:
1.  **Deconstruction & Identification:** First, parse the protein description. Identify all explicitly provided information, such as gene/protein name, source organism and strain, functional classification, virulence factor family, etc.
2.  **Expert Inference & Expansion:** This is the most critical step. Leverage your built-in bioinformatics knowledge base to reason and expand upon the information identified in the previous step. For instance, upon seeing "PscP" and "Pseudomonas aeruginosa," you must automatically connect this to concepts like "Type III Secretion System (TTSS)," "needle complex," and "effector protein translocation." You are expected to infer the protein's likely key conserved domains, subcellular localization, structural features, and its specific role in the virulence mechanism.
3.  **Prompt Synthesis:** Based on the complete analysis and inference, synthesize all information to construct the final prompt.

### Requirements for the Generated Prompt ###
* **Expert Tone:** It must be written in the voice of a scientist guiding a protein design project.
* **Comprehensive Content:** It must detail the protein's biological context, including but not limited to:
    * The biological significance of the source organism.
    * Its core function and mechanism of action as a virulence factor.
    * All inferred key domains, functional motifs, and anticipated 3D structural characteristics.
    * Its role within its biological pathway or protein complex.
* **Key Constraint:** Strictly adhere to the input requirements. The generated prompt must **not** contain any database IDs or accession numbers (e.g., VFG..., gb|..., WP_...). The focus must be solely on its biological and chemical essence.

### Input Data ###
{source_content}

### Output Format ###
Your final output must be a strict JSON object containing only a single key, `"prompt"`, whose value is the final prompt string you synthesized. Do not include any Markdown formatting, explanations, or any other extraneous text.
"""
# ==============================================================================


# --- 初始化异步API客户端 ---
try:
    # 从环境变量中获取API密钥并初始化DeepSeek的异步客户端
    aclient = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
except Exception as e:
    print(f"初始化OpenAI客户端时发生错误: {e}")
    # 如果初始化失败，将客户端设置为空，后续流程将安全退出
    aclient = None


def parse_jsonl(file_path: str, field_name: str) -> list[dict]:
    """
    解析JSONL文件，返回一个包含记录的列表。
    每个列表元素是一个字典，格式为 {'original_record': dict, 'content': str}。
    'original_record' 保存了原始的整行JSON对象，'content' 是从指定字段提取的文本内容。
    """
    records = []
    print(f"开始解析JSONL文件: '{file_path}'，提取字段: '{field_name}'")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行

                try:
                    # 将每一行文本解析为JSON对象
                    json_obj = json.loads(line)
                    # 使用 .get() 方法安全地提取指定字段的内容
                    content = json_obj.get(field_name)

                    if content is not None:
                        # 如果字段存在且内容不为空，则添加到记录列表中
                        records.append({
                            'original_record': json_obj,
                            'content': str(content) # 确保内容是字符串格式
                        })
                    else:
                        # 如果指定字段不存在，打印警告
                        print(f"警告: 第 {line_num} 行找不到字段 '{field_name}'。该行将被跳过。")

                except json.JSONDecodeError:
                    # 如果某一行不是有效的JSON格式，打印错误
                    print(f"错误: 第 {line_num} 行JSON解析失败。该行将被跳过。内容: '{line}'")

    except FileNotFoundError:
        print(f"错误: JSONL输入文件 '{file_path}' 未找到。")
        return []
    return records


async def get_llm_completion_async(prompt_text: str) -> str | None:
    """
    异步调用大语言模型API，获取完整的回答字符串。
    如果API客户端未成功初始化，则直接返回None。
    """
    if not aclient:
        return None
    try:
        response = await aclient.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            stream=False,       # 非流式传输，一次性获取完整结果
            temperature=0.2,    # 温度较低，使输出更具确定性
            # 关键：要求模型必须返回JSON对象，这可以减少后续处理的麻烦
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        # 捕获API调用过程中可能发生的任何异常
        return f"API_ERROR: {e}"


async def process_jsonl_record(record_data: dict, semaphore: asyncio.Semaphore, success_outfile, failure_outfile):
    """
    处理单个JSONL记录的完整异步任务单元。
    包括构建prompt、调用API、重试、验证结果和写入文件。
    """
    # 从传入的数据中解包原始记录和提取的内容
    original_record = record_data.get("original_record")
    source_content = record_data.get("content")

    if not source_content:
        # 如果主要内容为空，则无法处理，直接返回失败
        return False

    # 使用模板和从文件中提取的内容，构建最终发送给模型的Prompt
    prompt_to_send = PROMPT_TEMPLATE.format(source_content=source_content)

    # 使用信号量来控制并发数量，防止同时发起过多请求
    async with semaphore:
        retries = 0
        is_successful = False
        last_llm_response_obj = None # 用于存储最后一次API的响应，无论成功失败

        # 进入重试循环，直到成功或达到最大重试次数
        while not is_successful and retries < MAX_RETRIES:
            if retries > 0:
                # 如果不是第一次尝试，则等待一段时间再重试
                await asyncio.sleep(RETRY_DELAY_SECONDS)

            response_str = await get_llm_completion_async(prompt_to_send)
            retries += 1
            
            # 检查API调用是否失败
            if not response_str or response_str.startswith("API_ERROR:"):
                last_llm_response_obj = {"error": "API call failed", "details": response_str}
                continue # 继续下一次重试

            # 尝试将返回的字符串解析为JSON
            try:
                llm_response_obj = json.loads(response_str)
                last_llm_response_obj = llm_response_obj
                
                # ★★★ 核心验证：检查返回的JSON是否为字典，且包含一个名为'prompt'的非空字段 ★★★
                if isinstance(llm_response_obj, dict) and llm_response_obj.get("prompt"):
                    is_successful = True
                else:
                    # 如果验证失败，在响应对象中添加一个错误说明
                    last_llm_response_obj['validation_error'] = "JSON response does not contain a valid 'prompt' field."

            except json.JSONDecodeError:
                # 如果模型返回的不是有效的JSON，记录错误
                last_llm_response_obj = {"error": "Model returned non-JSON", "raw_response": response_str}
                continue # 继续下一次重试
        
        # 根据处理结果写入不同的文件
        if is_successful:
            # 成功：构建最终记录，包含原始数据和新生成的prompt
            final_record = {
                "original_data": original_record,
                "generated_prompt_for_sequence_model": last_llm_response_obj.get("prompt")
            }
            # 异步写入成功文件
            await success_outfile.write(json.dumps(final_record, ensure_ascii=False) + '\n')
            return True
        else:
            # 失败：记录原始数据、发送的prompt和失败的输岀
            failure_record = {
                "source_jsonl_record": original_record,
                "prompt_sent_to_model": prompt_to_send,
                "failed_model_output": last_llm_response_obj,
            }
            # 异步写入失败文件
            await failure_outfile.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
            return False


async def main():
    """主异步执行函数，协调整个流程"""
    print("启动JSONL源数据Prompt生成流程...")
    
    # 1. 解析JSONL文件
    all_records = parse_jsonl(INPUT_FILE, SOURCE_FIELD_NAME)
    if not all_records:
        print("没有可处理的记录，程序退出。")
        return
    
    total_records = len(all_records)
    print(f"从文件 '{INPUT_FILE}' 中成功解析出 {total_records} 条有效记录。")

    # 2. 创建并发控制器
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # 3. 异步打开输出文件并执行所有处理任务
    async with aiofiles.open(SUCCESS_OUTPUT_FILE, 'w', encoding='utf-8') as success_f, \
             aiofiles.open(FAILURE_OUTPUT_FILE, 'w', encoding='utf-8') as failure_f:
        
        # 为每条记录创建一个处理任务
        tasks = [process_jsonl_record(record, semaphore, success_f, failure_f) for record in all_records]
        
        # 使用tqdm显示进度条，并等待所有任务完成
        results = await asyncio_tqdm.gather(*tasks, desc="处理进度")

        # 4. 统计并打印最终结果
        successful_count = sum(1 for r in results if r is True)
        failed_count = len(results) - successful_count

        print("\n========================================")
        print("所有处理已完成！")
        print(f"✅ 成功生成Prompt: {successful_count} 条")
        print(f"❌ 处理失败或出错 (已记录): {failed_count} 条")
        print(f"最终数据集已保存到: '{SUCCESS_OUTPUT_FILE}'")
        if failed_count > 0:
            print(f"失败日志已保存到: '{FAILURE_OUTPUT_FILE}'")
        print("========================================")

if __name__ == "__main__":
    # 运行主异步函数
    asyncio.run(main())
