import tarfile
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# =================================================================================
# 这是一个真正的解决方案，而不是你那种“分步解压”的白日梦
# =================================================================================

def prepare_aishell3_dataset(
    tar_path: str, 
    content_path: str, 
    output_path: str
):
    """
    流式处理 AISHELL-3 的 .tar.gz 压缩包，将其转换为“一个 wav 配一个 txt”的格式。
    这个脚本的核心思想是避免一次性解压整个 36GB 的文件，从而在任何垃圾环境
    （比如你的 Colab）中都能运行。

    参数:
        tar_path (str): train.tar.gz 文件的路径。
        content_path (str): 从压缩包里单独解压出来的 content.txt 的路径。
        output_path (str): 处理完成后，数据存放的目标文件夹。
    """
    print("开始进行流式数据预处理... 这会很慢，但不会把你的硬盘撑爆。")
    
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. 预先加载标注文件到内存 ---
    # 这是我们唯一需要完整加载的东西。把它变成一个字典，查询起来会快得像闪电。
    # 这叫哈希表，是你这种水平的人需要学习的基础数据结构。
    print("正在加载标注文件 content.txt 到内存...")
    transcript_dict = {}
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 格式: SSB00050001.wav|今天 天气 不错 啊|j in1 t ian1...
                wav_name, text, pinyin = line.strip().split('|')
                # 我们只需要文件名和文本
                transcript_dict[wav_name.strip()] = text.strip()
            except ValueError:
                print(f"警告: 跳过格式错误的行: {line.strip()}")
                continue
    print(f"标注文件加载完毕，共找到 {len(transcript_dict)} 条记录。")

    # --- 2. 流式读取 .tar.gz 文件 ---
    # tarfile 模块允许我们逐个访问压缩包里的成员，而不需要把它们全部解压。
    # 这就是“流式处理”的精髓。
    print(f"正在打开压缩包: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        # 我们用 getmembers() 获取所有成员的列表，然后用 tqdm 显示进度条
        members = tar.getmembers()
        for member in tqdm(members, desc="正在处理文件"):
            # 我们只关心 .wav 文件
            if member.isfile() and member.name.endswith(".wav"):
                # 从成员路径中提取纯文件名，比如 'SSB00050001.wav'
                wav_filename = Path(member.name).name
                
                # 在我们预加载的字典里查找对应的文本
                text = transcript_dict.get(wav_filename)
                
                if text:
                    # 如果找到了文本，说明这是一个有效的训练样本
                    
                    # 1. 创建对应的 .txt 文件
                    txt_filename = wav_filename.replace(".wav", ".txt")
                    txt_filepath = output_dir / txt_filename
                    with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
                        f_txt.write(text)
                        
                    # 2. **单独**解压这个 .wav 文件到目标目录
                    # extract() 方法可以只解压单个成员。
                    # 我们先创建一个临时对象来重命名文件，因为原始文件在压缩包里有很长的路径
                    member.name = Path(member.name).name # 去掉原始路径，只保留文件名
                    tar.extract(member, path=output_dir)

    print("="*50)
    print("数据预处理完成！")
    print(f"所有转换后的数据都已保存在: {output_dir}")
    print("现在，你可以在你的训练脚本里，把 data_path 指向这个文件夹了。")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AISHELL-3 数据集流式预处理脚本")
    
    # Colab 环境里，文件路径通常是固定的，但好的习惯是把它们做成参数
    parser.add_argument(
        "--tar_path", 
        type=str, 
        default="./train.tar.gz",
        help="原始 train.tar.gz 文件的路径"
    )
    parser.add_argument(
        "--content_path", 
        type=str, 
        default="./AISHELL-3/train/content.txt",
        help="单独解压出来的 content.txt 文件的路径"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./AISHELL-3-processed",
        help="处理完成后，数据的输出路径"
    )
    
    args = parser.parse_args()
    
    prepare_aishell3_dataset(args.tar_path, args.content_path, args.output_path)
