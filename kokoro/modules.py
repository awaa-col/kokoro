# https://github.com/yl4579/StyleTTS2/blob/main/models.py
from .istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from transformers import AlbertModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Embedding, Linear, ModuleList
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from mamba_ssm import Mamba
import math
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from . import commons
from . import attentions


# 【v4.5 终极修复】恢复被意外删除的 DurationEncoder 类
class DurationEncoder(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = attentions.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = attentions.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + g
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 gin_channels=0,
                 use_mamba=False,  # 【v4.3 核心改造】
                 mamba_config=None # 【v4.3 核心改造】
                 ):
        super().__init__()
        self.n_vocab = n_vocab
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.use_mamba = use_mamba

        self.emb = nn.Embedding(n_vocab, in_channels)
        nn.init.normal_(self.emb.weight, 0.0, in_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        
        # 【v4.3 核心改造】在这里进行器官移植
        if self.use_mamba:
            print("TextEncoder 正在使用 Mamba 模块进行韵律预测。")
            if mamba_config is None:
                raise ValueError("Mamba config must be provided when use_mamba is True.")
            
            d_mamba_model = hidden_channels + gin_channels
            self.duration_predictor = Mamba(
                d_model=d_mamba_model,
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand']
            )
            # 【v4.5 终极修复】为 Mamba 的高维输出增加一个线性投影层，以降维到1
            self.mamba_proj = nn.Linear(d_mamba_model, 1)
        else:
            print("TextEncoder 正在使用 LSTM 模块进行韵律预测。")
            self.duration_predictor = DurationEncoder(
                hidden_channels, filter_channels, kernel_size, p_dropout, gin_channels=gin_channels)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.in_channels)  # [b, t, h]
        x = x.transpose(1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask, g=g)
        
        # 准备送入韵律预测模块的数据
        x_dp = x
        if g is not None:
            g_dp = g.clone()

        # 【v4.3 核心改造】根据模式选择不同的路径
        if self.use_mamba:
            # Mamba 需要 (B, L, D) 格式
            x_dp = x_dp.transpose(1, 2)
            if g is not None:
                style_expanded = g_dp.transpose(1, 2).expand(-1, x_dp.size(1), -1)
                x_dp = torch.cat([x_dp, style_expanded], dim=-1)
            
            x_dp.masked_fill_(x_mask.transpose(1, 2) == 0, 0.0)
            
            # 【v4.5 终极修复】使用 mamba_proj 进行降维打击
            mamba_out = self.duration_predictor(x_dp) # 输出是 (B, L, D)
            logw = self.mamba_proj(mamba_out).squeeze(-1) # 降维到 (B, L, 1) 再 squeeze 成 (B, L)
        else:
            # DurationEncoder 内部已经包含了 proj 层，所以它的输出已经是 (B, 1, L)
            logw = self.duration_predictor(x_dp, x_mask, g=g_dp).squeeze(1)

        stats = self.proj(x * x_mask)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        return m, logs, logw, x_mask


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, d_model, d_hid, nlayers, dropout=0.1, style_dim=128, use_mamba=False, mamba_config=None):
        super().__init__()
        self.text_encoder = TextEncoder(
            n_vocab=35, 
            in_channels=d_model, 
            out_channels=d_model, 
            hidden_channels=d_hid, 
            filter_channels=d_hid*4, 
            n_heads=2, 
            n_layers=4, 
            kernel_size=5, 
            p_dropout=0.1, 
            gin_channels=style_dim,
            use_mamba=use_mamba,
            mamba_config=mamba_config
        )
        self.F0Ntrain = F0Network_N(d_model=d_hid, out_dim=2, nlayers=nlayers, dropout=dropout, style_dim=style_dim)
        self.lstm = nn.LSTM(d_hid, d_hid//2, num_layers=1, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(d_hid, 1)

    def forward(self, x, s, text_lengths, m):
        m, logs, logw, x_mask = self.text_encoder(x, text_lengths, g=s)
        # some legacy code
        x = m.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        duration = self.duration_proj(x.transpose(1, 2)).squeeze(-1)
        return m, logs, duration, x_mask


# https://github.com/yl4579/StyleTTS2/blob/main/Utils/PLBERT/util.py
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state
