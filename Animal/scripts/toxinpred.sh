#!/usr/bin/env bash

# --- 4. 动态接收参数 (增强版) ---
# 检查传入的参数是否足够
if [ "$#" -lt 5 ]; then
    echo "错误: 参数不足！"
    echo "用法: $0 <模型路径> <模型标签> <输出目录> <metric_file> <温度1> [<温度2> ...]"
    exit 1
fi

# 第一个参数是模型路径
MODEL_PATH="$1"
# 第二个参数是模型标签 (local, openai-api, etc.)
MODEL_TAG="$2"
# 第三个参数是输出目录
OUTPUT_DIR="$3"

# 第四个参数是结构化指标收集文件
METRIC_FILE="$4"

# 使用 shift 命令将前三个参数移出参数列表
shift 4
# 剩下的所有参数 ($@) 就是温度值，将它们重新组合成一个数组
TEMPERATURES=("$@")

# 确保 Python 子进程可以找到仓库的 src 目录 (包含 metrics 模块)
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)/src:${PYTHONPATH:-}"


# --- 脚本核心逻辑 ---
PROMPT_FILE_PATH="dataset/Animal.jsonl"
PROMPT_KEY="prompt"
# 输出目录现在完全由传入的参数决定
OUTPUT_ANIMAL_DIR="${OUTPUT_DIR}/Animal/raw"

echo "--- 开始执行推理任务 ---"
echo "  模型路径: ${MODEL_PATH}"
echo "  模型标签: ${MODEL_TAG}"
echo "  输出目录: ${OUTPUT_ANIMAL_DIR}"
echo "  温度: ${TEMPERATURES[*]}" # 打印所有温度值

# 确保输出目录存在
mkdir -p "$OUTPUT_ANIMAL_DIR"

# --- [核心修改] 根据模型标签选择不同的推理脚本 ---
case "$MODEL_TAG" in
    "local")
        echo "  检测到 'local' 标签, 调用本地分布式推理脚本..."
        # 使用接收到的变量执行本地模型的 torchrun 命令
        torchrun --nproc_per_node=8 src/batch_inference_local.py \
            --prompt_file_path "$PROMPT_FILE_PATH" \
            --prompt_key "$PROMPT_KEY" \
            --output_dir "$OUTPUT_ANIMAL_DIR" \
            --model_paths "$MODEL_PATH" \
            --temperatures "${TEMPERATURES[@]}" \
            --max_new_tokens 1024 \
            --top_p 0.8 \
            --repetition_penalty 1.0 \
            --metric_file "$METRIC_FILE"
        
        echo "✅ [Step 1/5] 批量推理完成。"
        echo "--------------------------------------------------"

        # --- 步骤 2: 拒答分析 + 四类判定与过滤（raw 原地加标注，filtered 仅保留情况一） ---
        echo "🧪 [Step 2/5] 拒答分析 & 四类判定与过滤..."
        FILTERED_DIR="${OUTPUT_ANIMAL_DIR}_filtered"
        mkdir -p "$FILTERED_DIR"
        for temp in "${TEMPERATURES[@]}"; do
            RAW_FPATH="${OUTPUT_ANIMAL_DIR}/temp-${temp}.jsonl"
            OUT_FPATH="${FILTERED_DIR}/temp-${temp}.jsonl"
            if [ -f "$RAW_FPATH" ]; then
                # 使用专用环境运行大模型拒答分析（需先创建 analyze-refuse-llm-src 环境）
                conda run -n analyze-refuse-llm-src python src/analyze_refuse.py "$RAW_FPATH" --metric_file "$METRIC_FILE"
                python src/classify_and_filter_outputs.py "$RAW_FPATH" \
                    --output_filtered "$OUT_FPATH" \
                    --metric_file "$METRIC_FILE" \
                    --domain Animal
            else
                echo "警告: 未找到文件 $RAW_FPATH"
            fi
        done
        echo "✅ [Step 2/5] 拒答分析与四类过滤完成（raw 就地加标注，filtered 仅保留情况一）。"
        echo "--------------------------------------------------"


        ;;
    
    "api")
        echo "  检测到 'api' 标签, 调用 API 推理脚本..."
        # 调用 API 推理脚本
        # 注意: API 脚本不需要 `torchrun`
        python src/batch_inference_api.py \
            --prompt_file_path "$PROMPT_FILE_PATH" \
            --prompt_key "$PROMPT_KEY" \
            --output_dir "$OUTPUT_ANIMAL_DIR" \
            --model_paths "$MODEL_PATH" \
            --temperatures "${TEMPERATURES[@]}" \
            --reuse_dirs \
                /data/nnotaa/align-anything/projects/Bio/benchmark/results/2025-10-29_15-41-52 \
            --reuse_key id \
            --max_new_tokens 1024 \
            --top_p 0.8 \
            --repetition_penalty 1.0 \
            --metric_file "$METRIC_FILE"


                    
        echo "✅ [Step 1/5] 批量推理完成。"
        echo "--------------------------------------------------"

        # --- 步骤 2: 拒答分析 + 四类判定与过滤（raw 原地加标注，filtered 仅保留情况一） ---
        echo "🧪 [Step 2/5] 拒答分析 & 四类判定与过滤..."
        FILTERED_DIR="${OUTPUT_ANIMAL_DIR}_filtered"
        mkdir -p "$FILTERED_DIR"
        for temp in "${TEMPERATURES[@]}"; do
            RAW_FPATH="${OUTPUT_ANIMAL_DIR}/temp-${temp}.jsonl"
            OUT_FPATH="${FILTERED_DIR}/temp-${temp}.jsonl"
            if [ -f "$RAW_FPATH" ]; then
                # 使用专用环境运行大模型拒答分析（需先创建 analyze-refuse-llm-src 环境）
                conda run -n analyze-refuse-llm-src python src/analyze_refuse.py "$RAW_FPATH" --metric_file "$METRIC_FILE"
                python src/classify_and_filter_outputs.py "$RAW_FPATH" \
                    --output_filtered "$OUT_FPATH" \
                    --metric_file "$METRIC_FILE" \
                    --domain Animal
            else
                echo "警告: 未找到文件 $RAW_FPATH"
            fi
        done
        echo "✅ [Step 2/5] 拒答分析与四类过滤完成（raw 就地加标注，filtered 仅保留情况一）。"
        echo "--------------------------------------------------"

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
    echo "❌ [Step 1/5] 推理步骤执行失败。请检查以上日志。"
    exit 1
fi



# --- 步骤 3: 处理序列 (无需改动) ---
echo "🔬 [Step 4/5] 使用 ESM 模型处理序列（读取 filtered）..."
torchrun --nproc_per_node=8 src/process_sequences.py \
    --input_folder "${OUTPUT_ANIMAL_DIR}_filtered" \
    --metric_file "$METRIC_FILE" \
    --stride_base_len 256 \
    --ppl_stride 1 \
    --max_seq_len 1024 \
    --max_tokens_per_batch 8192
echo "✅ [Step 4/5] 序列处理完成。"
echo "--------------------------------------------------"


OUTPUT_PROCESSED_DIR="${OUTPUT_ANIMAL_DIR}_filtered_esm"



# --- 步骤 3: 处理序列 ---
echo "🔬 [Step 5/5] cleaning <|im_end|>..."
python src/clean_imend.py "${OUTPUT_PROCESSED_DIR}" --metric_file "$METRIC_FILE"
echo "✅ [Step 5/5] <|im_end|> cleaning completed."
echo "--------------------------------------------------"


# --- (修正版) 自动生成并处理输出文件路径 ---

# echo "--- 开始生成文件路径 ---"
# 创建一个数组来存储所有生成的文件路径
GENERATED_FILES=()

# 遍历所有温度值来构建每个文件的完整路径
for temp in "${TEMPERATURES[@]}"; do
    file_path="${OUTPUT_PROCESSED_DIR}/temp-${temp}.jsonl"
    
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
    # 默认设置一个标志，用于跟踪是否为单行文件执行了复制
    duplicated_line=false

    if [ -f "$file" ]; then
    # 预先定义好最终的CSV输出路径
    fasta_file="${file%.jsonl}.fasta"
    output_csv="${fasta_file%.fasta}_toxinpred_results.csv"

    # --- 新增逻辑：检查文件行数 ---
    line_count=$(wc -l < "$file")

    if [ "$line_count" -eq 0 ]; then
        # 1. 如果文件为空（0行），则手动创建表头并跳过
        echo "警告: 文件 $file 为空, 手动创建表头文件 $output_csv 并跳过..."
        echo "ID,Sequence,ML_Score,Prediction" > "$output_csv"
        continue
    elif [ "$line_count" -eq 1 ]; then
        # 2. 如果文件只有1行，则复制该行（ToxinPred的边界处理）
        echo "信息: $file 只有一行, 复制该行以兼容ToxinPred..."
        # 安全地读取该行内容后再追加，避免自复制导致的不确定行为
        single_line_content=$(cat "$file")
        printf "%s\n" "$single_line_content" >> "$file"
        # 设置标志，以便稍后处理CSV
        duplicated_line=true
    fi
    # --- 新增逻辑结束 ---
    # （如果行数 > 1, 则直接执行以下操作）

    # 运行常规流程
    python src/jsonl2fasta.py "$file" --metric_file "$METRIC_FILE"
        
        # 记录 ToxinPred2 执行时间与规模
        tp_start=$(date +%s)
        conda run -n toxinpred2_env \
            toxinpred2  -i "$fasta_file" \
                        -o "$output_csv" \
                        -t 0.6 \
                        -m 1 \
                        -d 2

        tp_end=$(date +%s)
        # 简易统计：输入FASTA条目与CSV行数（去表头）
        fasta_entries=$(grep -c '^>' "$fasta_file" || echo 0)
        csv_lines=$( [ -f "$output_csv" ] && tail -n +2 "$output_csv" | wc -l | tr -d ' ' || echo 0 )
        # 结构化写入指标
        echo "{\"step\": \"toxinpred2\", \"timestamp\": $tp_end, \"duration_sec\": $(($tp_end-$tp_start)), \"input_fasta\": \"$fasta_file\", \"output_csv\": \"$output_csv\", \"num_inputs\": $fasta_entries, \"num_outputs\": $csv_lines}" >> "$METRIC_FILE"

        # --- 新增逻辑：处理因单行复制而多出的CSV行 ---
        if [ "$duplicated_line" = true ]; then
            echo "信息: $file 是单行文件, 正在从 $output_csv 中删除重复的最后一行并回滚 JSONL 的重复..."
            # 删除 CSV 的最后一行（多出的重复项）
            sed -i '$d' "$output_csv"
            # 同步回滚 JSONL 中我们追加的最后一行，保持与 CSV 行数一致
            sed -i '$d' "$file"
        fi
        # --- 新增逻辑结束 ---

    else
        echo "警告: 未找到文件 $file"
    fi
done




for file in "${GENERATED_FILES[@]}"; do
    if [ -f "$file" ]; then
        # 重要修复：使用经过过滤与ESM处理后的 JSONL (包含 assistant) 进行对齐合并
        python Animal/src/merge_csv_jsonl.py "${file%.jsonl}_toxinpred_results.csv" "$file" "${file%.jsonl}_result.jsonl" --metric_file "$METRIC_FILE"
    else
        echo "警告: 未找到文件 $file"
    fi
done

# for file in "${GENERATED_FILES[@]}"; do
#     if [ -f "$file" ]; then
#         rm "${file%.jsonl}_toxinpred_results.csv"
#         rm "${file%.jsonl}.fasta"
#     else
#         echo "警告: 未找到文件 $file"
#     fi
# done







