# src/modules/model_v12.py

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import constants
from config.default_config import (
    NUM_TOTAL_MEG_CHANNELS, WINDOW_T, CONV_DIM, ATTN_DIM, LSTM_LAYERS, BI_DIRECTIONAL,
    DROPOUT_RATE, NUM_PHONEME_CLASSES
)


# ---------- v12 EEG2Rep + BiLSTM (+ CLS MHA) ----------

class EEG2RepInputEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, reduced_time_steps: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 125 -> ~62
            nn.Conv1d(128, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(reduced_time_steps)
        )
        self.embedding_dim = embedding_dim
        self.reduced_time_steps = reduced_time_steps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)  # (B, E, T')

class ChannelSelfAttention(nn.Module):
    def __init__(self, channels: int, attn_dim: int, dropout: float):
        super().__init__()
        self.d = attn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(1, attn_dim)
        self.k_proj = nn.Linear(1, attn_dim)
        self.v_proj = nn.Linear(1, attn_dim)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,C,T)
        B, C, T = x.size()
        x4 = x.permute(0, 2, 1).unsqueeze(-1)  # (B,T,C,1)
        Q, K, V = [proj(x4) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        Q, K, V = [t.reshape(B * T, C, self.d) for t in (Q, K, V)]
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d)
        wts = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.bmm(wts, V).mean(dim=1).view(B, T, self.d)  # (B,T,d)
        return out, wts.view(B, T, C, C)

class GatedFusion(nn.Module):
    def __init__(self, in_dims: List[int], fusion_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, fusion_dim), nn.ReLU(), nn.Dropout(dropout)) for d in in_dims])
        self.gate = nn.Sequential(nn.Linear(len(in_dims), 16), nn.ReLU(), nn.Linear(16, len(in_dims)))
        self.fusion_dim = fusion_dim
    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = [p(f) for p, f in zip(self.proj, feats)]  # list (B,Df)
        Z = torch.stack(Z, dim=1)                      # (B,K,Df)
        s = Z.mean(dim=-1)                             # (B,K)
        g = torch.softmax(self.gate(s), dim=1).unsqueeze(-1)  # (B,K,1)
        fused = (g * Z).sum(dim=1)                     # (B,Df)
        return fused, g.squeeze(-1)

class PhonemeModelV12(nn.Module):
    def __init__(self, conv_dim=CONV_DIM, attn_dim=ATTN_DIM, lstm_layers=LSTM_LAYERS, bi=BI_DIRECTIONAL, dropout=DROPOUT_RATE):
        super().__init__()
        hidden_dim = conv_dim * (2 if bi else 1)
        self.eeg2rep = EEG2RepInputEncoder(NUM_TOTAL_MEG_CHANNELS, conv_dim, WINDOW_T // 2)
        eeg2rep_feat_dim = conv_dim * (WINDOW_T // 2)

        self.conv = nn.Conv1d(NUM_TOTAL_MEG_CHANNELS, conv_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(conv_dim)
        self.drop = nn.Dropout(dropout)

        self.chan_attn = ChannelSelfAttention(NUM_TOTAL_MEG_CHANNELS, attn_dim, dropout)
        self.temporal_in = nn.Linear(conv_dim + attn_dim, conv_dim)

        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=conv_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bi, dropout=dropout if lstm_layers > 1 else 0.0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_pool = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout)
        nn.init.xavier_uniform_(self.cls_token)

        self.fusion = GatedFusion([eeg2rep_feat_dim, hidden_dim], fusion_dim=256, dropout=dropout)
        self.classifier = nn.Linear(256, NUM_PHONEME_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        f1 = self.eeg2rep(x).view(B, -1)

        y = self.drop(self.bn(self.conv(x))).permute(0, 2, 1)    # (B,T,C')
        attn_f, _ = self.chan_attn(x)                           # (B,T,A)
        seq_in = self.temporal_in(torch.cat([y, attn_f], dim=-1))
        seq_out, _ = self.lstm(seq_in)                          # (B,T,H)

        q = self.cls_token.expand(B, -1, -1)                    # (B,1,H)
        cls_out, _ = self.cls_pool(q, seq_out, seq_out)         # (B,1,H)
        f2 = cls_out.squeeze(1)

        fused, _ = self.fusion([f1, f2])
        return self.classifier(fused)
