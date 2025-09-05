import argparse
import torch
import torchaudio
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from mamba_ssm import Mamba
from typing import List
from pathlib import Path
from tqdm import tqdm
import shutil
import random

# --- 导入 Kokoro 核心组件 ---
from kokoro.model import KModel
from kokoro.pipeline import KPipeline
from kokoro.modules import DurationEncoder

# ================================================================
# 核心逻辑区
# ================================================================

def prepare_aishell3_dataset(config: dict, split: str):
    """
    【v3.0】从一个完整解压的 AISHELL-3 目录创建训练集或测试集。
    """
    if split == 'train':
        unzipped_dir = Path(config['paths']['aishell3_unzipped_train_dir'])
        output_path = Path(config['paths']['processed_data'])
    elif split == 'test':
        unzipped_dir = Path(config['paths']['aishell3_unzipped_test_dir'])
        output_path = Path(config['paths']['processed_data_test'])
    else:
        raise ValueError(f"未知的数据划分: {split}")

    content_path = unzipped_dir / "content.txt"
    wav_root_path = unzipped_dir / "wav"
    
    print("="*50, f"\n开始从已解压目录进行 {split} 数据预处理...")
    output_path.mkdir(exist_ok=True, parents=True)
    
    if not unzipped_dir.exists():
        print(f"错误: 找不到 AISHELL-3 的 {split} 目录: {unzipped_dir}")
        return False
    
    if not content_path.exists():
        print(f"错误: 在 {unzipped_dir} 中找不到标注文件 content.txt")
        return False
        
    transcript_dict = {}
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 【v3.1 修复】统一使用空格分割的解析逻辑，因为它已被证明对 test 集有效。
                # 之前的 | 分割逻辑是错误的，导致 train 集加载失败。
                parts = line.strip().split()
                if len(parts) < 2:  # 至少要有一个文件名和一个字
                    continue
                
                wav_name = parts[0]
                text = " ".join(parts[1:])
                
                transcript_dict[wav_name.strip()] = text.strip()
            except Exception: continue
    
    print(f"标注文件加载完毕，共 {len(transcript_dict)} 条记录。")

    all_wav_files = list(wav_root_path.glob("*/*.wav"))
    for src_wav_path in tqdm(all_wav_files, desc=f"正在整理 {split} 文件"):
        wav_filename = src_wav_path.name
        text = transcript_dict.get(wav_filename)
        
        if text:
            txt_filename = wav_filename.replace(".wav", ".txt")
            txt_filepath = output_path / txt_filename
            with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
                f_txt.write(text)
            
            dest_wav_path = output_path / wav_filename
            shutil.copy(src_wav_path, dest_wav_path)

    print("="*50, f"\n{split} 数据预处理完成！\n", "="*50)
    return True


class CustomAudioDataset(Dataset):
    def __init__(self, data_path: str, target_sr: int, max_samples: int = None):
        self.data_path = Path(data_path)
        self.wav_files = list(self.data_path.glob("*.wav"))
        if max_samples is not None:
            self.wav_files = self.wav_files[:max_samples]
        self.target_sr = target_sr
        # 移除了 resampler_cache，因为我们现在使用预处理好的数据
        # self.resampler_cache = {}

    def __len__(self): return len(self.wav_files)
    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        text_path = wav_path.with_suffix(".txt")
        if not text_path.exists(): return None
        with open(text_path, 'r', encoding='utf-8') as f: text = f.read().strip()
        
        # 直接加载已经重采样好的音频
        wav, sr = torchaudio.load(wav_path)
        
        # 【重要】做一个断言检查，确保我们加载的数据确实是正确的采样率
        # 这样可以避免因为数据目录错误导致的潜在问题
        assert sr == self.target_sr, f"音频采样率与目标不符！文件: {wav_path}, 实际: {sr}, 目标: {self.target_sr}"
        
        return {"text": text, "wav": wav.squeeze(0)}

class Collator:
    def __init__(self, pipeline: KPipeline): self.pipeline = pipeline
    def __call__(self, batch: List[dict]):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        texts, wavs = [i['text'] for i in batch], [i['wav'] for i in batch]
        phonemes_list = [self.pipeline.g2p(t)[0] for t in texts]
        input_ids_list = [torch.LongTensor([0, *list(filter(None, map(self.pipeline.model.vocab.get, p))), 0]) for p in phonemes_list]
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        mel_targets = [self.pipeline.model.stft.mel_spectrogram(w.unsqueeze(0)) for w in wavs]
        mel_padded = pad_sequence([m.squeeze(0).T for m in mel_targets], batch_first=True).transpose(1,2)
        ref_s_list = [self.pipeline.model.audio_to_ref_s(w) for w in wavs]
        ref_s = torch.stack(ref_s_list)
        return {"input_ids": input_ids_padded, "ref_s": ref_s, "mel_targets": mel_padded}

def replace_lstm_with_mamba_in_durationencoder(duration_encoder: DurationEncoder, mamba_config: dict):
    sty_dim, d_model = duration_encoder.sty_dim, duration_encoder.d_model
    mamba_module = Mamba(d_model=d_model + sty_dim, d_state=mamba_config['d_state'], d_conv=mamba_config['d_conv'], expand=mamba_config['expand'])
    duration_encoder.lstms = torch.nn.ModuleList([mamba_module])
    def mamba_forward(self, x, style, text_lengths, m):
        s = style.expand(x.shape[0], x.shape[1], -1)
        mamba_input = torch.cat([x.permute(2, 0, 1), s], axis=-1).transpose(0, 1)
        mamba_input.masked_fill_(m.unsqueeze(-1), 0.0)
        return self.lstms[0](mamba_input).transpose(-1, -2)
    DurationEncoder.forward = mamba_forward
    print("模型手术成功！DurationEncoder 已被 Mamba 强化。")
    return duration_encoder

def adapt_model_for_finetuning(model: KModel):
    original_forward = model.forward_with_tokens
    def training_forward(self, input_ids, ref_s, speed=1, return_mel=False):
        if not return_mel: return original_forward(input_ids, ref_s, speed)
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
        text_mask = torch.gt(torch.arange(input_lengths.max(), device=self.device).unsqueeze(0) + 1, input_lengths.unsqueeze(1))
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        max_len = pred_dur.sum(axis=-1).max()
        pred_aln_trg_list = []
        for i in range(input_ids.shape[0]):
            indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur[i])
            pred_aln_trg = torch.zeros((input_ids.shape[1], max_len), device=self.device)
            valid_indices = indices[indices < input_ids.shape[1]]
            pred_aln_trg[valid_indices, torch.arange(len(valid_indices), device=self.device)] = 1
            pred_aln_trg_list.append(pred_aln_trg)
        pred_aln_trg = torch.stack(pred_aln_trg_list)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        return t_en @ pred_aln_trg, F0_pred, N_pred
    KModel.forward_with_tokens = training_forward
    return model

def setup_peft(model: KModel):
    for param in model.parameters(): param.requires_grad = False
    for param in model.predictor.text_encoder.lstms[0].parameters(): param.requires_grad = True
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PEFT 设置完成. 可训练参数: {trainable_params} ({trainable_params/sum(p.numel() for p in model.parameters())*100:.4f}%)")
    return model
    
def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}, AMP: {'启用' if config['training']['use_amp'] else '禁用'}")
    
    model = KModel(repo_id=config['paths']['repo_id'])
    pipeline = KPipeline(lang_code='z', model=model, repo_id=config['paths']['repo_id'])
    
    model.predictor.text_encoder = replace_lstm_with_mamba_in_durationencoder(model.predictor.text_encoder, config['mamba'])
    if config['paths']['load_mamba_from']:
        try:
            model.predictor.text_encoder.lstms[0].load_state_dict(torch.load(config['paths']['load_mamba_from'], map_location=device))
            print(f"Mamba 权重从 {config['paths']['load_mamba_from']} 加载成功！")
        except Exception as e:
            print(f"加载 Mamba 权重失败: {e}")
    model = adapt_model_for_finetuning(model)
    model = setup_peft(model)
    model.to(device)
    
    # 【核心修改】现在我们从重采样后的目录加载数据
    final_data_path = Path(config['paths']['processed_data_resampled'])
    if not final_data_path.exists() or not any(final_data_path.iterdir()):
        print("="*50)
        print(f"错误: 最终训练数据目录 '{final_data_path}' 为空或不存在。")
        print("请按以下步骤操作:")
        print("  1. 运行 `python train.py --config config.yaml --prepare-data-only` 来生成中间数据。")
        print("  2. 运行 `python preprocess_audio.py --config config.yaml` 来对音频进行重采样。")
        print("程序将终止。")
        print("="*50)
        return

    print(f"将从最终预处理目录加载训练数据: {final_data_path}")
    dataset = CustomAudioDataset(final_data_path, target_sr=config['data']['target_sample_rate'])
    
    if len(dataset) == 0:
        print("="*50)
        print("错误：训练数据集为空！")
        print(f"请确保 '{final_data_path}' 目录中包含有效的 .wav 和 .txt 文件。")
        print("这通常是因为数据预处理步骤未能成功生成任何有效的训练样本。")
        print("请检查以下几点：")
        print(f"  1. 你的 `config.yaml` 中的 `aishell3_unzipped_train_dir` 路径 ('{config['paths']['aishell3_unzipped_train_dir']}') 是否正确。")
        print(f"  2. 运行 `preprocess_audio.py` 脚本是否成功，并且没有报错。")
        print("程序将终止。")
        print("="*50)
        return

    # 【新增】加载验证集
    final_test_data_path = Path(config['paths']['processed_data_test_resampled'])
    if not final_test_data_path.exists() or not any(final_test_data_path.iterdir()):
        print(f"警告: 最终验证数据目录 '{final_test_data_path}' 为空或不存在。将继续进行无验证的训练。")
        val_dataloader = None
    else:
        print(f"将从最终预处理目录加载验证数据: {final_test_data_path}")
        val_dataset = CustomAudioDataset(final_test_data_path, target_sr=config['data']['target_sample_rate'])
        if len(val_dataset) == 0:
            print(f"警告: 验证数据集为空，路径: {final_test_data_path}。将继续进行无验证的训练。")
            val_dataloader = None
        else:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config['training']['batch_size'], # 可以为验证集设置不同的 batch_size
                shuffle=False,
                collate_fn=collator,
                num_workers=config['training']['num_workers'],
                pin_memory=True
            )
            # 【新增】为验证音频样本准备固定数据
            sample_generation_batch = next(iter(val_dataloader))
            sample_output_dir = Path(config['paths']['output_dir']) / "samples"
            sample_output_dir.mkdir(exist_ok=True, parents=True)
            
    collator = Collator(pipeline)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=collator, 
        num_workers=config['training']['num_workers'], 
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(config['training']['use_amp'] and device.type == 'cuda'))
    
    print("="*50, "\n训练开始！\n", "="*50)
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [训练]")
        
        for batch in progress_bar:
            if batch is None: continue
            optimizer.zero_grad(set_to_none=True)
            
            input_ids, ref_s, mel_targets = batch['input_ids'].to(device), batch['ref_s'].to(device), batch['mel_targets'].to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(config['training']['use_amp'] and device.type == 'cuda')):
                predicted_mel, _, _ = model.forward_with_tokens(input_ids, ref_s, return_mel=True)
                min_len = min(predicted_mel.shape[-1], mel_targets.shape[-1])
                loss = loss_fn(predicted_mel[..., :min_len], mel_targets[..., :min_len])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_item = loss.item()
            train_loss += loss_item
            progress_bar.set_postfix({"loss": f"{loss_item:.4f}"})
            
        avg_train_loss = train_loss / len(dataloader)
        print(f"Epoch {epoch+1} 训练完成，平均损失: {avg_train_loss:.4f}")

        # 【新增】验证环节
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [验证]")
            with torch.no_grad():
                for batch in val_progress_bar:
                    if batch is None: continue
                    input_ids, ref_s, mel_targets = batch['input_ids'].to(device), batch['ref_s'].to(device), batch['mel_targets'].to(device)
                    
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(config['training']['use_amp'] and device.type == 'cuda')):
                        predicted_mel, _, _ = model.forward_with_tokens(input_ids, ref_s, return_mel=True)
                        min_len = min(predicted_mel.shape[-1], mel_targets.shape[-1])
                        loss = loss_fn(predicted_mel[..., :min_len], mel_targets[..., :min_len])
                    
                    val_loss += loss.item()
                    val_progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1} 验证完成，平均损失: {avg_val_loss:.4f}")

            # 【新增】生成并保存音频样本
            print("正在生成音频样本...")
            sample_input_ids = sample_generation_batch['input_ids'].to(device)
            sample_ref_s = sample_generation_batch['ref_s'].to(device)
            
            with torch.no_grad():
                # 只生成第一个样本以节省时间
                output_wav = model.forward_with_tokens(sample_input_ids[:1], sample_ref_s[:1])
            
            sample_path = sample_output_dir / f"epoch_{epoch+1}.wav"
            torchaudio.save(sample_path, output_wav.cpu(), config['data']['target_sample_rate'])
            print(f"音频样本已保存至: {sample_path}")

        if (epoch + 1) % config['training']['save_every_n_epochs'] == 0:
            output_path = Path(config['paths']['output_dir'])
            output_path.mkdir(exist_ok=True)
            mamba_weights = model.predictor.text_encoder.lstms[0].state_dict()
            save_path = output_path / f"mamba_peft_epoch_{epoch+1}.pth"
            torch.save(mamba_weights, save_path)
            print(f"Mamba 模块权重已保存: {save_path}")
            
    print("微调完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配置文件驱动的 Kokoro-Mamba PEFT 微调框架")
    parser.add_argument('--config', type=str, required=True, help='指向 config.yaml 文件的路径')
    parser.add_argument('--prepare-data-only', action='store_true', help='如果设置，则只运行数据准备步骤然后退出')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件: {args.config}")
        exit(1)

    if args.prepare_data_only:
        print("模式: 仅准备数据...")
        prepare_aishell3_dataset(config, 'train')
        prepare_aishell3_dataset(config, 'test')
        print("数据准备完成。")
        exit(0)
        
    train(config)
