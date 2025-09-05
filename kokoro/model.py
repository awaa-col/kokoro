from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch
from omegaconf import OmegaConf


class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        hps: OmegaConf, # 【v4.5】构造函数大修，现在只接收一个完整的 hps 对象
        model_path: Optional[str] = None # 允许外部传入模型路径
    ):
        super().__init__()
        self.hps = hps
        
        # 使用 hps 构建模型
        self.bert = torch.nn.Linear(hps.model.hidden_channels, hps.model.hidden_channels) # Placeholder, weights will be loaded
        self.bert_encoder = torch.nn.Linear(hps.model.hidden_channels, hps.model.hidden_channels)
        
        self.predictor = ProsodyPredictor(
            d_model=hps.model.hidden_channels, 
            d_hid=hps.model.hidden_channels, 
            nlayers=2, 
            style_dim=hps.model.style_dim,
            use_mamba=hps.train.get('use_mamba', False),      
            mamba_config=hps.train.get('mamba_config', None)
        )
        self.vocab = {i[1]:i[0] for i in hps.data.p_phonemes}
        self.p_phonemes = hps.data.p_phonemes
        
        # 加载预训练权重
        if not model_path:
            model_path = hf_hub_download(repo_id=hps.repo_id, filename=hps.model_filename)
        
        print(f"正在从 {model_path} 加载预训练权重...")
        state_dict = torch.load(model_path, map_location='cpu')['model']
        
        # 手动加载权重以处理不匹配和缺失的键
        self.load_state_dict(state_dict, strict=False)
        print("预训练权重加载完毕。")

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
        text_mask = torch.gt(torch.arange(input_lengths.max(), device=self.device).unsqueeze(0) + 1, input_lengths.unsqueeze(1))
        
        # 【v4.5】彻底重写前向传播逻辑
        # 注意：原始代码中的 `bert` 和 `bert_encoder` 在这里没有被直接使用，而是通过 `predictor` 内部的逻辑
        # 为了保持一致性，我们直接调用 predictor
        
        # 1. 通过 ProsodyPredictor 获取核心输出
        # ProsodyPredictor 的输入应该是 phoneme embeddings, style vector, lengths, mask
        # 但原始代码的 adapt_model_for_finetuning 逻辑非常混乱，我们尝试简化它
        # 假设 ref_s 是 style vector `s`
        s = ref_s
        m, logs, duration, x_mask = self.predictor(input_ids, s, input_lengths, text_mask)

        # 2. 根据预测的时长，创建对齐矩阵
        pred_dur = torch.round(duration).clamp(min=1).long()
        max_len = pred_dur.sum(axis=-1).max()
        pred_aln_trg_list = []
        for i in range(input_ids.shape[0]):
            pred_aln_trg_list.append(torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur[i]))
        pred_aln_trg = torch.stack(pred_aln_trg_list)

        # 3. 使用对齐矩阵扩展编码器输出
        # 这里的 `m` 相当于原始逻辑中的 `d` 或 `en`
        en = m @ pred_aln_trg

        # 4. F0 和 N 预测
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # 5. 解码器生成音频
        # 原始代码在这里又调用了一次 self.text_encoder, 这是冗余且错误的
        # 我们应该直接使用已经对齐的 en
        audio = self.decoder(en, F0_pred, N_pred, s).squeeze()
        
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
