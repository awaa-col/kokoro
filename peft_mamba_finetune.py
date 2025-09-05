import argparse
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from mamba_ssm import Mamba
from dataclasses import dataclass, field
from typing import List
import os
from pathlib import Path
from tqdm import tqdm
import tarfile
import random # <-- 新增：为了随机抽样

# --- 导入 Kokoro 核心组件 ---
# 别问我为什么这么写，这是为了让脚本在项目根目录能直接跑
# 如果你的路径有问题，自己想办法解决
from kokoro.kokoro.model import KModel
from kokoro.kokoro.pipeline import KPipeline
from kokoro.kokoro.modules import DurationEncoder

# =================================================================================
# 1. 配置参数 (Configuration)
# =================================================================================
# 用 dataclass 来管理超参数，比你用一堆散装变量要优雅一万倍
@dataclass
class TrainingConfig:
    """
    这不是普通的变量，这是“单一事实来源”(Single Source of Truth)。
    把所有能调的参数都放这里，让你的实验管理不再是一坨屎。
    """
    repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh"
    
    # 原始 AISHELL-3 数据路径
    aishell3_tar_path: str = "./train.tar.gz" 
    aishell3_content_path: str = "./AISHELL-3/train/content.txt"
    
    # data_path 现在指向处理后的数据
    data_path: str = "./AISHELL-3-processed"

    # 解除封印！我们现在有资源了，就要用全部的数据！
    sampling_ratio: float = 1.0 # <-- *** 已更新 ***: Pro 用户就该用 100% 的数据！
    
    # 训练参数
    epochs: int = 10 # <-- *** 已更新 ***: 根据 A100 的实际算力，调整到一个更合理的轮数
    batch_size: int = 16 # <-- *** 已更新 ***: A100/T4 就该用更大的 batch size!
    learning_rate: float = 1e-4
    
    # Mamba 配置 (这里的参数你可以随便玩，玩炸了别怪我)
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # 其他
    save_every_n_epochs: int = 5
    output_dir: str = "./kokoro_finetuned_model"

    # 新增：用于加载预训练 Mamba 模型的路径
    load_mamba_path: str = None # <-- *** 新增 ***: 第二阶段微调时指定

# =================================================================================
# 新增：数据预处理模块
# =================================================================================
def prepare_aishell3_dataset(
    tar_path: str, 
    content_path: str, 
    output_path: str,
    sampling_ratio: float = 1.0 # <-- *** 新增 ***
):
    """
    流式处理 AISHELL-3 的 .tar.gz 压缩包，将其转换为“一个 wav 配一个 txt”的格式。
    这个脚本的核心思想是避免一次性解压整个 36GB 的文件。
    """
    print("="*50)
    print("开始进行流式数据预处理... 这会很慢，但不会把你的硬盘撑爆。")
    
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    print("正在加载标注文件 content.txt 到内存...")
    transcript_dict = {}
    if not Path(content_path).exists():
        print(f"错误: 找不到 content.txt 文件: {content_path}")
        print("请先从 train.tar.gz 中单独解压出这个文件。")
        print("命令: tar -xvf train.tar.gz AISHELL-3/train/content.txt")
        return
        
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                wav_name, text, _ = line.strip().split('|', 2)
                transcript_dict[wav_name.strip()] = text.strip()
            except ValueError:
                continue
    print(f"标注文件加载完毕，共找到 {len(transcript_dict)} 条记录。")
    if sampling_ratio < 1.0:
        print(f"警告：将只处理 {sampling_ratio*100:.0f}% 的数据。")

    print(f"正在打开压缩包: {tar_path}")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="正在处理文件"):
                # 新增：按比例随机跳过文件
                if random.random() > sampling_ratio:
                    continue

                if member.isfile() and member.name.endswith(".wav"):
                    wav_filename = Path(member.name).name
                    text = transcript_dict.get(wav_filename)
                    
                    if text:
                        txt_filename = wav_filename.replace(".wav", ".txt")
                        txt_filepath = output_dir / txt_filename
                        with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
                            f_txt.write(text)
                            
                        member.name = Path(member.name).name
                        tar.extract(member, path=output_dir)
    except FileNotFoundError:
        print(f"错误: 找不到 train.tar.gz 文件: {tar_path}")
        print("请确保已下载 AISHELL-3 数据集。")
        return

    print("="*50)
    print("数据预处理完成！")
    print(f"所有转换后的数据都已保存在: {output_dir}")
    print("="*50)


# =================================================================================
# 2. 数据集处理 (Dataset & Collate)
# =================================================================================
# 你以为 PyTorch 的 Dataset 只是个可有可无的封装？
# 错了，这是构建可复现、高性能数据管道的基石。
class CustomAudioDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据集路径不存在，你是不是傻: {data_path}")
            
        self.wav_files = list(self.data_path.glob("*.wav"))
        if not self.wav_files:
            raise FileNotFoundError(f"在 {data_path} 没找到任何 .wav 文件。你的数据集呢？")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        text_path = wav_path.with_suffix(".txt")

        if not text_path.exists():
            # 如果对应的 .txt 文件不存在，直接跳过这个样本
            print(f"警告: 找不到 {wav_path} 对应的文本文件 {text_path}，跳过。")
            return None

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        wav, sr = torchaudio.load(wav_path)
        
        # 将音频重采样到 24kHz，这是 Kokoro 期望的采样率
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
            wav = resampler(wav)

        return {"text": text, "wav": wav.squeeze(0)}

# collate_fn 的作用是把一个 batch 里的数据“捏”成模型能吃的形状。
# 尤其是处理变长序列（比如你的音频和文本），没它你寸步难行。
class Collator:
    def __init__(self, pipeline: KPipeline):
        self.pipeline = pipeline

    def __call__(self, batch: List[dict]):
        # 过滤掉数据加载中可能返回的 None
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        texts = [item['text'] for item in batch]
        wavs = [item['wav'] for item in batch]
        
        # 1. 文本预处理 -> phonemes -> input_ids
        phonemes_list = [self.pipeline.text_to_phonemes(text) for text in texts]
        input_ids_list = [
            torch.LongTensor([0, *list(filter(None, map(self.pipeline.model.vocab.get, p))), 0])
            for p in phonemes_list
        ]
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        
        # 2. 音频预处理 -> 提取目标声学特征 (mel-spectrogram)
        # 这就是我们要用来计算损失的 "Ground Truth"
        mel_targets = [self.pipeline.stft.mel_spectrogram(wav.unsqueeze(0)) for wav in wavs]
        mel_padded = pad_sequence([m.squeeze(0).T for m in mel_targets], batch_first=True).transpose(1,2)
        
        # 3. 提取风格向量 ref_s
        ref_s_list = [self.pipeline.audio_to_ref_s(wav) for wav in wavs]
        ref_s = torch.stack(ref_s_list)
        
        return {
            "input_ids": input_ids_padded,
            "ref_s": ref_s,
            "mel_targets": mel_padded,
        }

# =================================================================================
# 3. 模型外科手术 (The Surgery)
# =================================================================================
# 这就是我说的“模型手术”。精确、高效，而不是你想象中的“玩炸了”。
def replace_lstm_with_mamba_in_durationencoder(
    duration_encoder: DurationEncoder, 
    config: TrainingConfig
) -> DurationEncoder:
    """
    找到那个叫 DurationEncoder 的老家伙，把它里面的 LSTM 扔了，换上 Mamba。
    """
    # 获取原始模块的参数
    sty_dim = duration_encoder.sty_dim
    d_model = duration_encoder.d_model
    
    print("正在执行模型手术：将 DurationEncoder 中的 LSTM 替换为 Mamba...")
    
    # 1. 创建新的 Mamba 模块
    # 注意 d_model 的维度，需要和原来 LSTM 的输入保持一致
    mamba_module = Mamba(
        d_model=d_model + sty_dim,
        d_state=config.mamba_d_state,
        d_conv=config.mamba_d_conv,
        expand=config.mamba_expand,
    )
    
    # 2. 替换掉原来的 lstms 列表
    duration_encoder.lstms = torch.nn.ModuleList([mamba_module]) # 用 mamba 替换
    
    # 3. 重写 forward 方法！这才是核心！
    # Mamba 不需要那些伺候 LSTM 的繁琐操作。
    def mamba_forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        
        # 准备 Mamba 的输入: [batch, length, dim]
        mamba_input = torch.cat([x, s], axis=-1).transpose(0, 1)
        mamba_input.masked_fill_(masks.unsqueeze(-1), 0.0)

        # 直接调用 Mamba！看到没有，多么简洁！
        mamba_output = self.lstms[0](mamba_input) # self.lstms[0] 现在就是 mamba
        
        # 确保输出维度和原来一致
        return mamba_output.transpose(-1, -2)

    # 用我们写好的新 forward 方法，替换掉原来的旧方法。这叫“猴子补丁”。
    DurationEncoder.forward = mamba_forward
    
    print("手术成功！DurationEncoder 已被 Mamba 强化。")
    return duration_encoder

# =================================================================================
# 4. 训练准备 (PEFT & Model Adaptation)
# =================================================================================
def adapt_model_for_finetuning(model: KModel) -> KModel:
    """
    修改模型的 forward_with_tokens 方法，让它在训练时返回声学特征而不是音频。
    这样我们才能计算损失。
    """
    # 保存原始方法，以备不时之需
    original_forward = model.forward_with_tokens

    def training_forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1, return_mel: bool = False):
        if not return_mel:
            return original_forward(input_ids, ref_s, speed)

        # --- 这里是原始 forward_with_tokens 的逻辑，但我们会在 decoder 前停下 ---
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0) # squeeze(0) for batch > 1
        
        # --- 创建对齐矩阵 ---
        # Note: 这部分在 batch > 1 时需要小心处理
        if input_ids.shape[0] > 1:
            # For simplicity, we just use the first item's duration for shape.
            # A more robust implementation would handle varied durations in a batch.
             max_len = pred_dur[0].sum()
             pred_aln_trg_list = []
             for i in range(input_ids.shape[0]):
                indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur[i])
                pred_aln_trg = torch.zeros((input_ids.shape[1], max_len), device=self.device)
                pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
                pred_aln_trg_list.append(pred_aln_trg)
             pred_aln_trg = torch.stack(pred_aln_trg_list)
        else:
            indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
            pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
            pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        
        # 这就是我们要的声学特征
        predicted_mel = t_en @ pred_aln_trg
        
        return predicted_mel, F0_pred, N_pred

    # 替换
    KModel.forward_with_tokens = training_forward
    print("模型 forward 方法已修改，以支持训练。")
    return model


def setup_peft(model: KModel):
    """
    冻结所有参数，然后只解冻我们新加的 Mamba 模块。
    这就是 PEFT 的精髓：花小钱，办大事。
    """
    for param in model.parameters():
        param.requires_grad = False
    
    print("模型所有参数已冻结。")

    # 精准解冻 Mamba 模块
    for param in model.predictor.text_encoder.lstms[0].parameters():
        param.requires_grad = True
        
    print("Mamba 模块已解冻，准备进行参数高效微调。")
    
    # 打印可训练参数数量，让你看看 PEFT 有多高效
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
    
    return model

# =================================================================================
# 5. 训练循环 (The Main Loop)
# =================================================================================
def train(config: TrainingConfig):
    # --- 新增：自动预处理数据 ---
    processed_data_dir = Path(config.data_path)
    if not processed_data_dir.exists() or not any(processed_data_dir.iterdir()):
        print("处理后的数据文件夹不存在或为空。")
        prepare_aishell3_dataset(
            config.aishell3_tar_path,
            config.aishell3_content_path,
            config.data_path,
            config.sampling_ratio # <-- *** 新增 ***
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # --- 1. 加载模型和 Pipeline ---
    print(f"正在从 {config.repo_id} 加载预训练模型...")
    model = KModel(repo_id=config.repo_id)
    pipeline = KPipeline(model=model)
    
    # --- 2. 执行手术和改造 ---
    model.predictor.text_encoder = replace_lstm_with_mamba_in_durationencoder(model.predictor.text_encoder, config)

    # --- 新增：加载 Mamba 预训练权重 ---
    if config.load_mamba_path:
        try:
            print(f"正在从 {config.load_mamba_path} 加载 Mamba 预训练权重...")
            mamba_weights = torch.load(config.load_mamba_path, map_location=device)
            model.predictor.text_encoder.lstms[0].load_state_dict(mamba_weights)
            print("Mamba 权重加载成功！")
        except Exception as e:
            print(f"加载 Mamba 权重失败，你给的路径是不是有问题: {e}")
            print("将从头开始训练 Mamba。")

    model = adapt_model_for_finetuning(model)
    model = setup_peft(model)
    model.to(device)
    
    # --- 3. 准备数据 ---
    print(f"正在从处理好的数据文件夹加载数据: {config.data_path}")
    dataset = CustomAudioDataset(config.data_path)
    collator = Collator(pipeline)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=8 # <-- *** 已更新 ***: Pro 用户值得拥有更快的 IO
    )
    
    # --- 4. 设置优化器和损失函数 ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate
    )
    # 我们用 L1 Loss (MAE) 来比较生成的声学特征和真实的声学特征
    loss_fn = torch.nn.L1Loss()
    
    # --- 5. 开始训练 ---
    print("="*50)
    print("一切准备就绪，开始微调！")
    print("="*50)
    
    for epoch in range(config.epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        total_loss = 0.0
        
        for batch in progress_bar:
            if batch is None:
                continue

            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            ref_s = batch['ref_s'].to(device)
            mel_targets = batch['mel_targets'].to(device)
            
            # 调用我们修改过的 forward 方法
            predicted_mel, _, _ = model.forward_with_tokens(
                input_ids, ref_s, return_mel=True
            )
            
            # 裁剪预测结果以匹配目标长度
            min_len = min(predicted_mel.shape[-1], mel_targets.shape[-1])
            predicted_mel = predicted_mel[..., :min_len]
            mel_targets = mel_targets[..., :min_len]

            loss = loss_fn(predicted_mel, mel_targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")

        # --- 6. 保存模型 ---
        if (epoch + 1) % config.save_every_n_epochs == 0:
            output_path = Path(config.output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 我们只保存 Mamba 模块的权重，因为其他部分都没变
            mamba_weights = model.predictor.text_encoder.lstms[0].state_dict()
            save_path = output_path / f"mamba_peft_epoch_{epoch+1}.pth"
            torch.save(mamba_weights, save_path)
            print(f"Mamba 模块权重已保存至: {save_path}")
            
    print("微调完成！你现在有了一个被 Mamba 强化过的 Kokoro 模型。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Mamba 对 Kokoro 模型进行 PEFT 微调")
    # 你可以在命令行里覆盖掉 dataclass 里的默认值
    # 比如: python peft_mamba_finetune.py --data_path /path/to/your/data --epochs 100
    config = TrainingConfig()
    # HACK: Use `getattr` to handle potential missing fields in older versions
    # of the config dataclass, ensuring backward compatibility.
    for f in config.__dataclass_fields__:
        # Get the default value from the dataclass definition
        default_val = config.__dataclass_fields__[f].default
        # Get the type of the default value for argparse
        field_type = type(default_val) if default_val is not None else str
        parser.add_argument(f'--{f}', type=field_type, default=default_val)
    
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    train(config)
