import csv
import sys

# --- 用户配置区 ---
# 请在此处修改您的文件名和列名

# 1. CSV 文件路径
csv_file_path = 'outfile.csv'

# 2. 包含ID的列名
id_column_name = 'ID'

# 3. 输入的 FASTA 文件路径
input_fasta_path = 'uniprot_animal_cdhit.fasta'

# 4. 输出的 FASTA 文件路径
output_fasta_path = 'true_positive.fasta'

# --- 脚本主代码区 ---

def load_ids_from_csv(file_path, column_name):
    """从CSV文件中加载ID到集合中以便快速查找。"""
    print(f"正在从 '{file_path}' 文件中加载 ID...")
    allowed_ids = set()
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if column_name not in reader.fieldnames:
                print(f"错误：CSV文件中未找到名为 '{column_name}' 的列。")
                print(f"可用的列有: {', '.join(reader.fieldnames)}")
                sys.exit(1) # 终止脚本
            
            for row in reader:
                # 确保行和列都存在
                if column_name in row and row[column_name]:
                    allowed_ids.add(row[column_name])
                    
    except FileNotFoundError:
        print(f"错误：找不到CSV文件 '{file_path}'。请检查文件名和路径是否正确。")
        sys.exit(1)
        
    if not allowed_ids:
        print("警告：未从CSV文件中加载任何ID。输出文件将为空。")
        
    print(f"加载完成。共找到 {len(allowed_ids)} 个唯一的 ID。")
    return allowed_ids

def filter_fasta_file(input_path, output_path, allowed_ids):
    """根据提供的ID集合过滤FASTA文件。"""
    print(f"\n正在过滤 FASTA 文件 '{input_path}'...")
    written_count = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            keep_entry = False
            for line in infile:
                if line.startswith('>'):
                    # 这是一个新的条目描述行
                    # 提取ID。默认假设ID是 '>' 之后到第一个空格之前的部分。
                    # 例如，从 '>ID123 gene=xyz' 中提取 'ID123'
                    # !!! 如果您的ID格式不同，请修改下面这一行 !!!
                    current_id = line.lstrip('>').split()[0]
                    
                    if current_id in allowed_ids:
                        keep_entry = True
                        written_count += 1
                        outfile.write(line)
                    else:
                        keep_entry = False
                elif keep_entry:
                    # 如果是序列行，并且该条目需要保留，则写入
                    outfile.write(line)

    except FileNotFoundError:
        print(f"错误：找不到输入的 FASTA 文件 '{input_path}'。请检查文件名和路径是否正确。")
        sys.exit(1)
        
    print(f"过滤完成。共写入 {written_count} 个条目到 '{output_path}'。")
    return written_count

if __name__ == "__main__":
    # 步骤 1: 加载ID
    ids_to_keep = load_ids_from_csv(csv_file_path, id_column_name)
    
    # 步骤 2: 过滤FASTA文件
    final_count = filter_fasta_file(input_fasta_path, output_fasta_path, ids_to_keep)
    
    # 步骤 3: 最终验证
    print("\n--- 验证 ---")
    if final_count == len(ids_to_keep):
        print("成功！输出的FASTA条目数与CSV文件中的ID数完全一致。")
    else:
        print(f"警告：输出的条目数 ({final_count}) 与CSV文件中的ID数 ({len(ids_to_keep)}) 不匹配。")
        print("这可能是因为FASTA文件中缺少某些ID，或者FASTA头部的ID格式与CSV中的不完全匹配。")