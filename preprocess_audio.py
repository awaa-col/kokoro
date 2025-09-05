import argparse
import yaml
from pathlib import Path
import torchaudio
from tqdm import tqdm
import shutil

def run_preprocessing_for_split(config: dict, split: str):
    """
    对指定的数据划分（'train' 或 'test'）进行音频预处理。
    """
    if split == 'train':
        source_key = 'processed_data'
        target_key = 'processed_data_resampled'
    elif split == 'test':
        source_key = 'processed_data_test'
        target_key = 'processed_data_test_resampled'
    else:
        raise ValueError(f"未知的数据划分: {split}")

    source_data_path = Path(config['paths'][source_key])
    target_data_path = Path(config['paths'][target_key])

    if not source_data_path.exists() or not any(source_data_path.iterdir()):
        print(f"警告: 原始 {split} 数据目录 '{source_data_path}' 为空或不存在，跳过处理。")
        return

    target_data_path.mkdir(exist_ok=True, parents=True)
    target_sr = config['data']['target_sample_rate']
    
    print("-" * 50)
    print(f"开始处理 '{split}' 数据集...")
    print(f"源目录: {source_data_path}")
    print(f"目标目录: {target_data_path}")
    print("-" * 50)

    # 复制所有 .txt 文件
    txt_files = list(source_data_path.glob("*.txt"))
    for txt_file in tqdm(txt_files, desc=f"复制 {split} .txt 文件"):
        shutil.copy(txt_file, target_data_path / txt_file.name)

    # 处理所有 .wav 文件
    wav_files = list(source_data_path.glob("*.wav"))
    resampler_cache = {}
    
    for wav_path in tqdm(wav_files, desc=f"处理 {split} .wav 文件"):
        wav, sr = torchaudio.load(wav_path)
        
        if sr != target_sr:
            if sr not in resampler_cache:
                resampler_cache[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler_cache[sr](wav)
        
        torchaudio.save(
            target_data_path / wav_path.name,
            wav,
            target_sr
        )
    print(f"'{split}' 数据集处理完成。")


def preprocess_audio_files(config: dict):
    """
    一次性将所有音频文件重采样到目标采样率，并复制相应的文本文件。
    这是一个独立的预处理步骤，旨在避免在训练循环中进行低效的实时重采样。
    """
    print("="*50)
    print(f"开始音频预处理...")
    print(f"目标采样率: {config['data']['target_sample_rate']} Hz")
    
    # 同时处理训练集和测试集
    run_preprocessing_for_split(config, 'train')
    run_preprocessing_for_split(config, 'test')
    
    print("="*50)
    print("所有音频预处理完成！")
    print(f"请检查以下目录以确认文件已生成:")
    print(f"  - 训练集: {config['paths']['processed_data_resampled']}")
    print(f"  - 测试集: {config['paths']['processed_data_test_resampled']}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro 音频预处理脚本")
    parser.add_argument('--config', type=str, required=True, help='指向 config.yaml 文件的路径')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件: {args.config}")
        exit(1)
        
    preprocess_audio_files(config)
