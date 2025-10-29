#!/usr/bin/env bash

# toxinpred2不能处理0条和1条序列的输入文件，会报错



# --- 1. 集中配置 ---
# 在这里定义所有要测试的模型及其标签 (格式: "模型路径或名称:标签")
# 标签可以是 'local', 'openai-api', 'google-api' 等，用于在子脚本中区分处理逻辑
MODEL_CONFIGS=(
    "grok-4:api"
    "claude-opus-4:api"
    "deepseek-chat:api"
    "gemini-2.5-pro:api"
    "gpt-5:api"
    "/data/nnotaa/align-anything/projects/Bio/outputs/Qwen2.5-7B-Instruct/slice_5380:local"
    "Qwen/Qwen2.5-7B-Instruct:local"
    # "gpt-4-turbo:openai-api"
    # "meta-llama/Llama-3-8B-Instruct:local"
    # "gemini-pro:google-api"
)
TEMPERATURES=(0.0 0.7) # 您可以传递多个温度值

# --- 基础设置 ---
# 切换到项目根目录，如果失败则退出
cd '/data/nnotaa/align-anything/projects/Bio/benchmark' || exit

# 确保所有子进程能够导入 repo 下的 src 包 (包含 metrics.py)
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BASE_RESULTS_DIR="results/${TIMESTAMP}"
# BASE_RESULTS_DIR="results/tst"

# 提前创建结果根目录并启动调试日志 Tee
mkdir -p "${BASE_RESULTS_DIR}"
LOG_FILE="${BASE_RESULTS_DIR}/benchmark_run.log"
# 将 stdout/stderr 同时输出到屏幕与日志文件
exec &> >(tee -a "${LOG_FILE}")

# --- 2. 循环调度 ---
# 遍历 MODEL_CONFIGS 数组中的每一个配置
for config in "${MODEL_CONFIGS[@]}"; do
    # --- 解析配置 ---
    # 使用':'作为分隔符，将配置分割为模型路径和标签
    IFS=':' read -r model_path model_tag <<< "$config"

    echo "🚀 开始处理模型: ${model_path} (标签: ${model_tag})"

    # --- 路径处理 ---
    # 将模型名称中的 '/' 替换为 '_'，为当前模型创建独立的文件夹名
    SANITIZED_MODEL_NAME="${model_path//\//_}"
    TARGET_PATH="${BASE_RESULTS_DIR}/${SANITIZED_MODEL_NAME}"
    METRIC_FILE="${TARGET_PATH}/benchmark_metrics.jsonl"

    echo "   - 清理后的模型名: ${SANITIZED_MODEL_NAME}"
    echo "   - 创建输出目录: ${TARGET_PATH}"

    # 确保目标输出目录存在
    mkdir -p "${TARGET_PATH}"

    # --- 3. 动态传递参数给子脚本 ---
    # 调用子脚本，并传递：
    # 参数1: 当前的模型路径 (一个字符串)
    # 参数2: 当前模型的输出目录 (一个字符串)
    # 参数3及以后: 整个 TEMPERATURES 数组的元素

    # 【修改 3】: 在调用子脚本时，增加 $model_tag 参数
    # 调用子脚本，并额外传递模型标签 (model_tag)
    
    echo "   - 正在运行 Bacteria exo prediction 任务..."
    bash Bacteria/scripts/exo.sh "$model_path" "$model_tag" "$TARGET_PATH" "$METRIC_FILE" "${TEMPERATURES[@]}"
    # bash Bacteria/scripts/exo.sh "$model_path" "$TARGET_PATH" "${TEMPERATURES[@]}"
    
    echo "   - 正在运行 Animal toxin prediction 任务..."
    bash Animal/scripts/toxinpred.sh "$model_path" "$model_tag" "$TARGET_PATH" "$METRIC_FILE" "${TEMPERATURES[@]}"
    # bash Animal/scripts/toxinpred.sh "$model_path" "$TARGET_PATH" "${TEMPERATURES[@]}"


    # 在模型流程末尾输出并保存汇总报告
    echo "📊 生成指标汇总报告..."
    python src/summarize_benchmark.py \
        --metric_file "$METRIC_FILE" \
        --results_dir "$TARGET_PATH" \
        --output_report "$TARGET_PATH/benchmark_summary.txt"

    echo "✅ 模型 ${model_path} 处理完成。"
    echo "-----------------------------------------"

done

echo "🎉 所有任务执行完毕！"