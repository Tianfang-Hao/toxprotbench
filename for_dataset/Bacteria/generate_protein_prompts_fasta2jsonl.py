import os
import json
import asyncio
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiofiles
from tqdm.asyncio import tqdm as asyncio_tqdm

# 在程序开始时加载 .env 文件中的环境变量
load_dotenv()

# ==============================================================================
# --- 配置 ---
# ==============================================================================
# 并发请求数，可以根据您的API速率限制进行调整
CONCURRENT_REQUESTS = 50
# 使用的模型名称
MODEL_NAME = "deepseek-chat"
# 输入的FASTA文件名
INPUT_FILE = 'VFDB_filtered.fasta'
# 成功处理结果的输出文件名
SUCCESS_OUTPUT_FILE = 'VFDB_detailed.jsonl'
# 处理失败记录的输出文件名
FAILURE_OUTPUT_FILE = 'protein_processing_failures.jsonl'
# API请求失败后的最大重试次数
MAX_RETRIES = 3
# 每次重试前的等待时间（秒）
RETRY_DELAY_SECONDS = 2

# ★★★ 在下面的三引号之间粘贴您的完整Prompt模板 ★★★
PROMPT_TEMPLATE = """
### Role ###
You are a top-tier Prompt Engineer with deep expertise in bioinformatics and protein engineering. Your knowledge base includes major protein, gene, and virulence factor databases (e.g., VFDB, UniProt, Pfam, GenBank).

### Context ###
I am evaluating a set of protein descriptions extracted from a bacterial virulence factor database. My goal is to determine if the information within each description is sufficient for an advanced, biologically-aware protein generation model to reproduce a sequence with the correct toxic function and key structural features.

### Task & Step-by-Step Instructions ###
For the "Protein Description" I provide, you must strictly follow these steps:

1.  **In-depth Analysis:**
    * **Extract Key Information:** Identify and list the following core biological information from the description:
        * Database IDs (e.g., VFG ID, GenBank ID)
        * Protein/Gene Name
        * Source Organism & Strain
        * Described Function/Category
        * Clues for Protein Family or Conserved Domains

2.  **Sufficiency Evaluation:**
    * **Core Evaluation Criteria:** Based on the analysis in Step 1, determine if the description is "sufficiently detailed." You must adhere strictly to the following criteria:
        * **Considered 'Sufficiently Detailed' (conclusion: true):** The description should be considered sufficient if the combination of **source organism, protein family, gene name, or functional keywords** allows you (as a bioinformatics expert) to **infer the protein's key functional domains, structural topology, or its specific biological pathway.** For example, even if domains are not explicitly listed, the combination of "PscP" and "Pseudomonas aeruginosa" is sufficient to infer its role as part of the TTSS needle complex.
        * **Considered 'Not Sufficiently Detailed' (conclusion: false):** The description should be considered insufficient if the information (even if it includes a species name) is too generic or vague to allow you to infer specific structural or functional information that would be instructive for sequence generation.

    * **State Conclusion:** In a single line, clearly state your evaluation conclusion, using either "The description is sufficiently detailed" or "The description is not sufficiently detailed."

### Input Data ###
{fasta_description}

### Output Format ###
Please structure your entire response strictly in the following Markdown format, without any additional explanations.

---
**1. In-depth Analysis**
* **Database IDs:** ...
* **Protein/Gene Name:** ...
* **Source Organism & Strain:** ...
* **Described Function/Category:** ...
* **Clues for Protein Family/Domains:** ...

**2. Evaluation Conclusion**
The description is sufficiently detailed / The description is not sufficiently detailed.

**3. JSON Object**
Output only a single JSON object. This object must contain only one key, "conclusion", with a boolean value (true or false) indicating if the content is sufficiently detailed.

If sufficiently detailed, output: {{"conclusion": true}}

If not sufficiently detailed, output: {{"conclusion": false}}"""
# ==============================================================================
# --- 脚本主体 ---
# ==============================================================================
# --- 初始化异步API客户端 ---
try:
    aclient = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY not found in .env file or environment variables.")
except Exception as e:
    print(f"初始化API客户端时发生错误: {e}")
    aclient = None


def parse_fasta(file_path: str) -> list[dict]:
    """
    解析FASTA文件，返回一个包含记录的列表。
    """
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = None
            sequence_parts = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        records.append({'header': header, 'sequence': ''.join(sequence_parts)})
                    header = line[1:]
                    sequence_parts = []
                else:
                    sequence_parts.append(line)
            if header is not None:
                records.append({'header': header, 'sequence': ''.join(sequence_parts)})
    except FileNotFoundError:
        print(f"错误: FASTA输入文件 '{file_path}' 未找到。请确保文件存在且路径正确。")
        return []
    return records

# ★★★ 核心修改：新的、更宽松的JSON提取函数 ★★★
def extract_json_from_text(text: str) -> str | None:
    """
    从文本中提取最后一个结构完整的JSON对象。
    该方法通过从后向前查找 '{' 并平衡括号来实现，非常稳健。
    """
    try:
        # 从字符串末尾找到最后一个开括号 '{'
        start_index = text.rfind('{')
        if start_index == -1:
            return None

        # 从该位置开始，平衡括号以找到完整的JSON对象
        brace_count = 0
        substring = text[start_index:]
        for i, char in enumerate(substring):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            
            if brace_count == 0:
                # 括号平衡，我们找到了一个完整的JSON块
                potential_json = substring[:i+1]
                # 尝试解析以确认它是有效的JSON
                json.loads(potential_json)
                return potential_json
        
        # 如果循环结束括号仍未平衡，则没有找到有效的JSON
        return None
    except (json.JSONDecodeError, IndexError):
        # 如果在任何步骤出现错误，则表示未找到
        return None


async def get_llm_completion_async(prompt_text: str) -> str | None:
    """异步调用API，获取完整的回答字符串。"""
    if not aclient: return None
    try:
        response = await aclient.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            stream=False,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API_ERROR: {e}"


async def process_fasta_record(record: dict, semaphore: asyncio.Semaphore, success_outfile, failure_outfile):
    """
    处理单个FASTA记录的完整异步任务单元。
    """
    fasta_header = record.get("header")
    protein_sequence = record.get("sequence")

    if not fasta_header or not protein_sequence:
        return False

    prompt_to_send = PROMPT_TEMPLATE.format(fasta_description=fasta_header)

    async with semaphore:
        retries = 0
        is_successful = False
        last_error_info = None

        while not is_successful and retries < MAX_RETRIES:
            if retries > 0:
                await asyncio.sleep(RETRY_DELAY_SECONDS)

            response_str = await get_llm_completion_async(prompt_to_send)
            retries += 1
            
            if not response_str or response_str.startswith("API_ERROR:"):
                last_error_info = {"error": "API call failed", "details": response_str}
                continue
            
            # ★★★ 使用新的函数来提取JSON ★★★
            json_str = extract_json_from_text(response_str)

            if not json_str:
                last_error_info = {"error": "Could not find any valid JSON object in the model response", "raw_response": response_str}
                continue

            try:
                llm_json_obj = json.loads(json_str) # 再次解析以在主逻辑中使用
                
                # 核心验证逻辑：检查'conclusion'键是否存在且其值为布尔型
                if isinstance(llm_json_obj, dict) and "conclusion" in llm_json_obj and isinstance(llm_json_obj.get("conclusion"), bool):
                    is_successful = True
                    final_record = {
                        "original_fasta_header": fasta_header,
                        "protein_sequence": protein_sequence,
                        "is_sufficient": llm_json_obj.get("conclusion"),
                        "llm_full_response": response_str 
                    }
                    await success_outfile.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                    return True
                else:
                    last_error_info = {"error": "JSON validation failed", "details": "The 'conclusion' key is missing or not a boolean.", "parsed_json": llm_json_obj, "raw_response": response_str}

            except json.JSONDecodeError:
                # 理论上新函数已经保证了可解析性，但作为安全措施保留
                last_error_info = {"error": "Failed to decode JSON from model response", "extracted_json_string": json_str, "raw_response": response_str}
        
        if not is_successful:
            failure_record = {
                "source_fasta_record": record,
                "prompt_sent_to_model": prompt_to_send,
                "last_attempt_error": last_error_info,
            }
            await failure_outfile.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
            return False


async def main():
    """主异步执行函数"""
    print("启动FASTA蛋白质描述处理流程...")
    
    all_records = parse_fasta(INPUT_FILE)
    if not all_records:
        return
    
    total_records = len(all_records)
    print(f"从文件 '{INPUT_FILE}' 中成功解析出 {total_records} 条蛋白质记录。")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    async with aiofiles.open(SUCCESS_OUTPUT_FILE, 'w', encoding='utf-8') as success_f, \
             aiofiles.open(FAILURE_OUTPUT_FILE, 'w', encoding='utf-8') as failure_f:
        
        tasks = [process_fasta_record(record, semaphore, success_f, failure_f) for record in all_records]
        results = await asyncio_tqdm.gather(*tasks, desc="处理进度")

        successful_count = sum(1 for r in results if r is True)
        failed_count = len(results) - successful_count

        print("\n========================================")
        print("所有处理已完成！")
        print(f"✅ 成功处理: {successful_count} 条")
        print(f"❌ 处理失败 (已记录): {failed_count} 条")
        print(f"成功结果已保存到: '{SUCCESS_OUTPUT_FILE}'")
        if failed_count > 0:
            print(f"失败日志已保存到: '{FAILURE_OUTPUT_FILE}'")
        print("========================================")

if __name__ == "__main__":
    if "请在这里粘贴您自己的完整PROMPT模板" in PROMPT_TEMPLATE:
         print("错误：请先将您的Prompt模板粘贴到脚本中的 PROMPT_TEMPLATE 变量处。")
    elif aclient:
        asyncio.run(main())
    else:
        print("API客户端未成功初始化，程序退出。请检查您的.env文件和API密钥。")