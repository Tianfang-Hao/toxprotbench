#!/usr/bin/env bash

# --- 4. 动态接收参数 (增强版) ---
# 检查传入的参数是否足够
if [ "$#" -lt 4 ]; then
    echo "错误: 参数不足！"
    echo "用法: $0 <模型路径> <模型标签> <输出目录> <温度1> [<温度2> ...]"
    exit 1
fi

# 第一个参数是模型路径
MODEL_PATH="$1"
# 第二个参数是模型标签 (local, openai-api, etc.)
MODEL_TAG="$2"
# 第三个参数是输出目录
OUTPUT_DIR="$3"

# 使用 shift 命令将前三个参数移出参数列表
shift 3
# 剩下的所有参数 ($@) 就是温度值，将它们重新组合成一个数组
TEMPERATURES=("$@")


# --- 脚本核心逻辑 ---
PROMPT_FILE_PATH="dataset/test.jsonl"
PROMPT_KEY="prompt"
# 输出目录现在完全由传入的参数决定
OUTPUT_BACTERIA_DIR="${OUTPUT_DIR}/Bacteria"

echo "--- 开始执行推理任务 ---"
echo "  模型路径: ${MODEL_PATH}"
echo "  模型标签: ${MODEL_TAG}"
echo "  输出目录: ${OUTPUT_BACTERIA_DIR}"
echo "  温度: ${TEMPERATURES[*]}" # 打印所有温度值

# 确保输出目录存在
mkdir -p "$OUTPUT_BACTERIA_DIR"

# --- [核心修改] 根据模型标签选择不同的推理脚本 ---
case "$MODEL_TAG" in
    "local")
        echo "  检测到 'local' 标签, 调用本地分布式推理脚本..."
        # 使用接收到的变量执行本地模型的 torchrun 命令
        torchrun --nproc_per_node=8 src/batch_inference_local.py \
            --prompt_file_path "$PROMPT_FILE_PATH" \
            --prompt_key "$PROMPT_KEY" \
            --output_dir "$OUTPUT_BACTERIA_DIR" \
            --model_paths "$MODEL_PATH" \
            --temperatures "${TEMPERATURES[@]}" \
            --max_new_tokens 1024 \
            --top_p 0.8 \
            --repetition_penalty 1.0
        ;;
    
    "openai-api")
        echo "  检测到 'openai-api' 标签, 调用 API 推理脚本..."
        # 调用 API 推理脚本
        # 注意: API 脚本不需要 `torchrun`
        python src/batch_inference_api.py \
            --prompt_file_path "$PROMPT_FILE_PATH" \
            --prompt_key "$PROMPT_KEY" \
            --output_dir "$OUTPUT_BACTERIA_DIR" \
            --model_paths "$MODEL_PATH" \
            --temperatures "${TEMPERATURES[@]}" \
            --max_new_tokens 1024 \
            --top_p 0.8 \
            --repetition_penalty 1.0
        ;;

    # --- 扩展点 ---
    # 如果将来有其他 API (例如 'gemini-api'), 在这里添加新的 case
    # "gemini-api")
    #     echo "  检测到 'gemini-api' 标签, 调用 Gemini API 推理脚本..."
    #     python src/batch_inference_gemini.py ...
    #     ;;

    *)
        echo "错误: 未知的模型标签 '$MODEL_TAG'。请在主脚本中检查 MODEL_CONFIGS。"
        exit 1
        ;;
esac

# 检查上一步命令是否成功执行
if [ $? -ne 0 ]; then
    echo "❌ [步骤 1/3] 推理步骤执行失败。请检查以上日志。"
    exit 1
fi

echo "✅ [步骤 1/3] 批量推理完成。"
echo "--------------------------------------------------"


# --- 步骤 2: 过滤结果 (无需改动) ---
echo "🧹 [步骤 2/3] 过滤结果..."
python src/filter_no_imend.py "$OUTPUT_BACTERIA_DIR"
echo "✅ [步骤 2/3] 过滤完成。结果位于 'filtered_$OUTPUT_BACTERIA_DIR'."
echo "--------------------------------------------------"


# --- 步骤 3: 处理序列 (无需改动) ---
echo "🔬 [步骤 3/3] 使用 ESM 模型处理序列..."
torchrun --nproc_per_node=8 src/process_sequences.py \
    --input_folder "${OUTPUT_BACTERIA_DIR}_filtered"
echo "✅ [步骤 3/3] 序列处理完成。"
echo "--------------------------------------------------"


OUTPUT_PROCESSED_DIR="${OUTPUT_BACTERIA_DIR}_filtered_esm"



# --- 步骤 3: 处理序列 ---
echo "🔬 [Step 4/3] cleaning <|im_end|>..."
python src/clean_imend.py "${OUTPUT_PROCESSED_DIR}"
echo "✅ [Step 4/3] <|im_end|> cleaning completed."
echo "--------------------------------------------------"


echo "--- 推理任务结束 ---"

# --- (修正版) 自动生成并处理输出文件路径 ---

# echo "--- 开始生成文件路径 ---"
# 创建一个数组来存储所有生成的文件路径
GENERATED_FILES=()

# 遍历所有温度值来构建每个文件的完整路径
for temp in "${TEMPERATURES[@]}"; do
    file_path="${OUTPUT_BACTERIA_DIR}/temp-${temp}.jsonl"
    
    # 将生成的文件路径添加到数组中
    GENERATED_FILES+=("$file_path")
    
    # echo "已生成路径: $file_path"
done

# echo "--- 所有文件路径已生成并存入 GENERATED_FILES 数组 ---"


# --- 后续使用示例 ---
# 你可以在这里对这些文件进行后续处理
# echo "所有文件的完整路径如下:"
# printf "%s\n" "${GENERATED_FILES[@]}"

for file in "${GENERATED_FILES[@]}"; do
    if [ -f "$file" ]; then
        python src/jsonl2fasta.py "$file"

    else
        echo "警告: 未找到文件 $file"
    fi
done


Bacteria/scripts/fasta2h5.sh "$OUTPUT_BACTERIA_DIR"


for file in "${GENERATED_FILES[@]}"; do
    if [ -f "$file" ]; then
        conda run -n ml python Bacteria/src/classifying_unknown_proteins.py Bacteria/Predictor/sklearn_svcPC20_SST30_CV10_embeddingsProtT5 "${file%.jsonl}_per_protein.h5" "${file%.jsonl}.fasta"
    else
        echo "警告: 未找到文件 $file"
    fi
done


for file in "${GENERATED_FILES[@]}"; do
    if [ -f "$file" ]; then
        python Bacteria/src/merge_jsonl.py "$PROMPT_FILE_PATH" "${file%.jsonl}_per_protein.jsonl" "${file%.jsonl}_result.jsonl"
    
    else
        echo "警告: 未找到文件 $file"
    fi
done


for file in "${GENERATED_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "${file%.jsonl}_per_protein.h5"
        rm "${file%.jsonl}_per_protein.jsonl"
        rm "${file%.jsonl}.fasta"
    else
        echo "警告: 未找到文件 $file"
    fi
done
