import os
import json
import argparse
import shutil   # 导入用于文件移动（替换）
import tempfile # 导入用于安全创建临时文件
from metrics import write_metric

def process_jsonl_file(filepath, field_name, token):
    """
    处理单个jsonl文件，删除指定字段末尾的特定标记，并安全地覆盖原文件。
    """
    
    # 1. 在原文件同一目录下创建一个临时文件
    #    'delete=False' 保证我们可以手动重命名它
    #    'dir=...' 确保临时文件和原文件在同一文件系统，使 'move' 成为原子操作
    try:
        temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                               dir=os.path.dirname(filepath), 
                                               delete=False)
        temp_filepath = temp_file.name
    except Exception as e:
        print(f"[错误] 无法在 {os.path.dirname(filepath)} 中创建临时文件: {e}")
        return

    print(f"--- 正在处理 (将覆盖): {filepath} ---")
    
    lines_processed = 0
    lines_modified = 0
    processing_error = False  # 标记是否发生错误

    try:
        # 2. 打开原文件进行读取，打开临时文件进行写入
        with open(filepath, 'r', encoding='utf-8') as infile, temp_file:
            
            for line in infile:
                lines_processed += 1
                try:
                    # 1. 解析json行
                    data = json.loads(line.strip())
                    
                    # 2. 检查字段是否存在且为字符串
                    if field_name in data and isinstance(data[field_name], str):
                        
                        # 3. 检查字符串是否以特定标记结尾
                        if data[field_name].endswith(token):
                            # 4. 删除结尾的标记
                            data[field_name] = data[field_name][:-len(token)]
                            lines_modified += 1
                    
                    # 5. 将处理后（或未修改）的数据写回 *临时* 文件
                    temp_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                except json.JSONDecodeError:
                    print(f"  [警告] 文件 {filepath} 第 {lines_processed} 行JSON解析失败，已跳过。")
                except Exception as e:
                    print(f"  [错误] 处理 {filepath} 第 {lines_processed} 行时发生意外错误: {e}")
                    processing_error = True  # 发生错误，标记一下
                    break # 停止处理此文件

    except FileNotFoundError:
        print(f"[错误] 文件未找到: {filepath}")
        processing_error = True
    except Exception as e:
        print(f"[严重错误] 无法读/写文件 {filepath}: {e}")
        processing_error = True

    # 3. 根据处理结果决定是替换还是保留原文件
    if not processing_error:
        # 没有任何错误，用临时文件替换原文件
        try:
            shutil.move(temp_filepath, filepath)
            print(f"处理完成: {filepath} (已覆盖)")
            print(f"  总行数: {lines_processed}")
            print(f"  修改行数: {lines_modified}\n")
            return lines_processed, lines_modified
        except Exception as e:
            print(f"[严重错误] 替换原文件 {filepath} 失败: {e}")
            # 尝试删除临时文件
            try:
                os.remove(temp_filepath)
            except OSError:
                pass # 忽略清理错误
    else:
        # 处理过程中出错，删除临时文件，保留原文件
        print(f"[操作取消] {filepath} 未被修改，因处理中发生错误。")
        try:
            os.remove(temp_filepath)
        except OSError as e:
            print(f"  [警告] 删除临时文件 {temp_filepath} 失败: {e}")
    return 0, 0


def main():
    """
    主函数，解析命令行参数并遍历文件夹。
    （此函数与您提供的版本相同，仅修正了缩进和可能的特殊空格）
    """
    # 1. 初始化参数解析器
    parser = argparse.ArgumentParser(
        description="批量清理 JSONL 文件中指定字段的末尾标记 (覆盖模式)。",
        # formatter_class 可以在显示help时保留换行符
        formatter_class=argparse.RawTextHelpFormatter 
    )
    
    # 2. 添加必需的参数 (位置参数)
    parser.add_argument(
        "directory",  # 参数名称
        type=str,
        help="包含 .jsonl 文件的目标文件夹路径。"
    )
    
    # 3. 修改 "field" 为可选参数（-f），并设置默认值
    parser.add_argument(
        "-f", "--field",      # 参数名称
        type=str,
        default="assistant",
        help="需要清理的JSON字段的名称 (例如: 'assistant', 'output')。\n(默认值: 'assistant')"
    )
    
    # 4. 添加可选的 token 参数
    parser.add_argument(
        "-t", "--token", # 短参数和长参数
        type=str,
        default="<|im_end|>", # 设置默认值
        help="需要从字段末尾移除的字符串。\n(默认值: '<|im_end|>')"
    )
    parser.add_argument("--metric_file", type=str, default=None, help="指标输出 JSONL 文件")
    
    # 5. 解析命令行传入的参数
    args = parser.parse_args()

    # 6. 使用解析到的参数
    directory_path = args.directory
    field_to_clean = args.field
    token_to_remove = args.token

    # 检查路径有效性
    if not os.path.isdir(directory_path):
        print(f"[错误] 路径无效或不是一个文件夹: {directory_path}")
        return

    # 打印任务信息
    print(f"*** 警告：脚本将直接覆盖原始文件，请确保已备份！ ***\n")
    print(f"开始批量处理文件夹: {directory_path}")
    print(f"目标字段: {field_to_clean}")
    print(f"移除标记: {token_to_remove}\n")
    
    file_count = 0
    # 遍历文件夹
    total_files = 0
    total_lines = 0
    total_modified = 0
    for filename in os.listdir(directory_path):
        # 确保只处理 .jsonl 文件，并跳过已清理过的文件
        # (您保留的这个 _cleaned.jsonl 检查逻辑可以防止重复处理旧脚本的输出)
        if filename.startswith('.'):
            # 跳过内部标记目录或隐藏文件（如 .tasks_temp-*.jsonl 目录）
            continue
        filepath = os.path.join(directory_path, filename)
        if os.path.isdir(filepath):
            # 严格跳过目录，即使其名以 .jsonl 结尾
            continue
        if filename.endswith(".jsonl") and not filename.endswith("_cleaned.jsonl"):
            file_count += 1
            # 7. 将参数传递给处理函数
            lp, lm = process_jsonl_file(filepath, field_to_clean, token_to_remove)
            total_files += 1
            total_lines += lp
            total_modified += lm
            
    if file_count == 0:
        print("在指定文件夹中未找到匹配的 .jsonl 文件。")
    else:
        print(f"批量处理完毕，共处理 {file_count} 个文件。")
    # 写指标
    write_metric(args.metric_file, "clean_imend", {
        "input_dir": directory_path,
        "input_files": total_files,
        "num_read": total_lines,
        "num_modified": total_modified,
    })

if __name__ == "__main__":
    main()