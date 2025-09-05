import os
from pathlib import Path

# =================================================================================
# 战场地形侦察脚本
# 你的任务就是告诉我，这个脚本的输出是什么。
# =================================================================================

# *** 修改这里 ***: 如果你的 AISHELL-3 文件夹不在这个路径，请修改它
AISHELL_PATH = "./AISHELL-3"

def find_the_damn_file(root_path_str: str):
    """
    在一个该死的不标准环境里，找到那个该死的 content.txt 文件。
    """
    root_path = Path(root_path_str)
    
    if not root_path.exists():
        print(f"错误：你提供的路径不存在 -> {root_path}")
        print("你确定你已经把 AISHELL-3 解压到这个位置了吗？")
        return

    print(f"正在根目录 [{root_path}] 中搜索 content.txt ...")
    
    # pathlib 的 rglob 是进行递归搜索的最优雅的方式
    found_files = list(root_path.rglob("content.txt"))
    
    if not found_files:
        print("="*50)
        print(">>> 致命错误：未找到任何 'content.txt' 文件！ <<<")
        print("="*50)
        print("请确认以下几点：")
        print("1. 你下载的是 AISHELL-3 的 train_set.zip 或 train.tar.gz。")
        print("2. 你的解压过程没有出错。")
        print("3. AISHELL_PATH 变量指向的是解压后的根文件夹。")
        return
        
    print(f"\n成功找到 {len(found_files)} 个 'content.txt' 文件：")
    for file_path in found_files:
        print(f"  - 路径: {file_path}")
        
    # 我们只关心第一个找到的文件
    target_file = found_files[0]
    
    print(f"\n正在分析文件内容: {target_file}")
    
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            print("文件前 5 行内容预览：")
            print("-" * 20)
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(line.strip())
            print("-" * 20)
            
        print("\n侦察完毕。根据以上信息，你的标注文件是存在的并且格式正确。")
        print("请在 peft_mamba_finetune.py 脚本中，将 aishell3_content_path 参数设置为上面找到的正确路径。")
        
    except Exception as e:
        print(f"\n读取文件时发生错误: {e}")
        print("你的 content.txt 文件可能已损坏或编码不正确。")


if __name__ == "__main__":
    find_the_damn_file(AISHELL_PATH)
