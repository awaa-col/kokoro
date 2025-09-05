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

# --- 导入 Kokoro 核心组件 ---
from kokoro.model import KModel
from kokoro.pipeline import KPipeline
from kokoro.modules import DurationEncoder

# ================================================================
# 核心逻辑区
# ================================================================

def prepare_aishell3_dataset_from_unzipped(config: dict):
    """
    【最终版 v2.0】从一个完整解压的 AISHELL-3 目录创建训练集。
    """
    # 【升级】现在从一个根目录读取路径
    unzipped_dir = Path(config['paths']['aishell3_unzipped_train_dir'])
    content_path = unzipped_dir / "content.txt"
    wav_root_path = unzipped_dir / "wav"
    output_path = Path(config['paths']['processed_data'])
    
    print("="*50, "\n开始从已解压目录进行数据预处理...")
    output_path.mkdir(exist_ok=True)
    
    if not unzipped_dir.exists():
        print(f"错误: 找不到 AISHELL-3 的训练目录: {unzipped_dir}")
        print("请确保你已经将 train.tar.gz 完整解压，并正确配置了 config.yaml 中的路径。")
        return False
        
    transcript_dict = {}
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                wav_name, text, _ = line.strip().split('|', 2)
                transcript_dict[wav_name.strip()] = text.strip()
            except ValueError: continue
    
    print(f"标注文件加载完毕，共 {len(transcript_dict)} 条记录。")

    all_wav_files = list(wav_root_path.glob("*/*.wav"))
    for src_wav_path in tqdm(all_wav_files, desc="正在整理文件"):
        wav_filename = src_wav_path.name
        text = transcript_dict.get(wav_filename)
        
        if text:
            txt_filename = wav_filename.replace(".wav", ".txt")
            txt_filepath = output_path / txt_filename
            with open(txt_filepath, 'w', encoding='utf-8') as f_txt:
                f_txt.write(text)
            
            dest_wav_path = output_path / wav_filename
            shutil.copy(src_wav_path, dest_wav_path)

    print("="*50, "\n数据预处理完成！\n", "="*50)
    return True

class CustomAudioDataset(Dataset):
    def __init__(self, data_path: str, target_sr: int):
        self.data_path = Path(data_path)
        self.wav_files = list(self.data_path.glob("*.wav"))
        self.target_sr = target_sr
        self.resampler_cache = {}

    def __len__(self): return len(self.wav_files)
    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        text_path = wav_path.with_suffix(".txt")
        if not text_path.exists(): return None
        with open(text_path, 'r', encoding='utf-8') as f: text = f.read().strip()
        wav, sr = torchaudio.load(wav_path)
        if sr != self.target_sr:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = self.resampler_cache[sr](wav)
        return {"text": text, "wav": wav.squeeze(0)}

class Collator:
    def __init__(self, pipeline: KPipeline): self.pipeline = pipeline
    def __call__(self, batch: List[dict]):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        texts, wavs = [i['text'] for i in batch], [i['wav'] for i in batch]
        phonemes_list = [self.pipeline.text_to_phonemes(t) for t in texts]
        input_ids_list = [torch.LongTensor([0, *list(filter(None, map(self.pipeline.model.vocab.get, p))), 0]) for p in phonemes_list]
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        mel_targets = [self.pipeline.stft.mel_spectrogram(w.unsqueeze(0)) for w in wavs]
        mel_padded = pad_sequence([m.squeeze(0).T for m in mel_targets], batch_first=True).transpose(1,2)
        ref_s_list = [self.pipeline.audio_to_ref_s(w) for w in wavs]
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
    if not (Path(config['paths']['processed_data']).exists() and any(Path(config['paths']['processed_data']).iterdir())):
        if not prepare_aishell3_dataset_from_unzipped(config): return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}, AMP: {'启用' if config['training']['use_amp'] else '禁用'}")
    
    model = KModel(repo_id=config['paths']['repo_id'])
    pipeline = KPipeline(model=model)
    
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
    
    dataset = CustomAudioDataset(config['paths']['processed_data'], target_sr=config['data']['target_sample_rate'])
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
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
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
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
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
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件: {args.config}")
        exit(1)
        
    train(config)
