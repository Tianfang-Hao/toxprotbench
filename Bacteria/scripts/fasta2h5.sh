#!/usr/bin/env bash

set -e

# --- 1. 用户配置区 ---
# 在这里定义您的输入路径和输出目录。
# 示例 1: 处理一个文件夹
# INPUT_PATH="fasta_input"

# 示例 2: 只处理一个文件
INPUT_PATH="$1"

OUTPUT_DIR="$1"
# 指标文件（可选第二参数）
METRIC_FILE="$2"
# --------------------
# conda activate align-anything

# 定义二级结构模型和示例文件的路径与URL
SEC_STRUCT_DIR="Bacteria/protT5/sec_struct_checkpoint"
SEC_STRUCT_URL="http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt"
EXAMPLE_FASTA_URL="https://raw.githubusercontent.com/agemagician/ProtTrans/refs/heads/master/example_seqs.fasta"
PYTHON_SCRIPT="Bacteria/src/fasta2h5.py"

# --- 步骤 1: 安装 Python 依赖 (如果需要) ---
echo "--- 步骤 1: 正在安装所需的 Python 库... ---"
pip install torch transformers sentencepiece h5py

# --- 步骤 2: 创建所有必要的目录 ---
echo -e "\n--- 步骤 2: 正在创建所有目录... ---"
mkdir -p "$SEC_STRUCT_DIR"
mkdir -p "$OUTPUT_DIR"
# 如果输入路径是目录且不存在，也创建它
if [[ ! "$INPUT_PATH" =~ \.fasta$ && ! -d "$INPUT_PATH" ]]; then
    mkdir -p "$INPUT_PATH"
fi
echo "目录已准备就绪。"

# --- 步骤 3: 检查并下载所需文件 ---
echo -e "\n--- 步骤 3: 正在检查并下载所需文件... ---"

# 检查并下载二级结构模型
if [ ! -f "$SEC_STRUCT_DIR/secstruct_checkpoint.pt" ]; then
    echo "未找到二级结构模型，正在下载..."
    wget -nc -P "$SEC_STRUCT_DIR" "$SEC_STRUCT_URL"
else
    echo "二级结构模型已存在，跳过下载。"
fi


# --- 步骤 4: 运行 Python 脚本并传递路径 ---
echo -e "\n--- 步骤 4: 环境准备就绪，正在启动 Python 脚本... ---"
# 调用Python脚本，并使用新的 --input_path 参数
python "$PYTHON_SCRIPT" --input_path "$INPUT_PATH" --output_dir "$OUTPUT_DIR" --metric_file "$METRIC_FILE"

echo -e "\n--- 所有操作执行完毕！结果保存在 '$OUTPUT_DIR' 目录中。 ---"