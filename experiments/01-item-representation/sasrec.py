#!/usr/bin/env python3
"""
SASRec 백본 + 교체 가능한(pluggable) 아이템 표현(item representation).

이 실험의 핵심 설계: 백본(self-attention 시퀀스 인코더)·학습 목적·평가를 모두
고정하고, "아이템을 어떻게 표현하는가"만 단일 변수로 바꾼다.

  - ItemRepresentation: 입력 lookup과 출력 scoring이 공유하는 (V+1, d_model) 행렬
    M을 생성한다. 행 0은 padding(전부 0). mode에 따라 M의 출처만 달라진다:
        'id'       -> 학습되는 nn.Embedding 테이블 (ID 기반 표현)
        'text'     -> 텍스트 임베딩(frozen 또는 train_text로 학습) + 학습되는 projection
        'randproj' -> text(train_text)의 용량 대조군: 동일 구조/파라미터지만 텍스트가
                      아닌 '무작위 초기화' 768차원 학습 테이블 + projection. 승리가
                      '의미적 초기화' 때문인지 '용량' 때문인지 분리하기 위함.
        'hybrid'   -> id + text 의 합
  - 입력 임베딩과 출력 점수(weight tying)가 동일한 M을 쓰므로, 두 실험 arm 사이에서
    바뀌는 것은 오직 표현뿐이다 -> apples-to-apples.

SASRec 구현 메모:
- 시퀀스는 오른쪽 패딩(right-pad)한다. 즉 위치 0은 항상 실제 아이템이다. (왼쪽
  패딩을 쓰면 위치 0이 pad가 되고, causal mask 하에서 그 쿼리는 자기 자신만 볼 수
  있는데 key_padding_mask가 그 키마저 가려 '모든 키가 masked'가 되어 softmax가
  NaN을 내고 풀링 벡터를 오염시킨다 -- 실제로 그 버그를 겪어 right-pad로 바꿨다.)
  사용자 표현 = 각 시퀀스의 '마지막 실제 위치'의 hidden state다(길이로 인덱싱).
- causal self-attention + key_padding_mask 로 패딩 토큰의 기여를 차단한다.
- 안정적 학습을 위해 pre-LN(norm_first=True) transformer 블록을 사용한다.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn


class ItemRepresentation(nn.Module):
    """입력 lookup과 출력 scoring이 공유하는 아이템 표현 행렬을 만든다.

    두 실험 arm 사이에서 바뀌는 유일한 요소다.
    """

    def __init__(self, mode, num_items, d_model, text_matrix=None,
                 train_text=False, latent_dim=768, dropout=0.0):
        super().__init__()
        self.mode = mode
        self.num_items = num_items            # 어휘 크기 V (인덱스 1..V, 0 = pad)
        self.d_model = d_model

        if mode in ("id", "hybrid"):
            # 학습되는 ID 임베딩 테이블. padding_idx=0 -> 행 0은 0으로 고정.
            self.id_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
            nn.init.normal_(self.id_emb.weight, std=0.02)
            with torch.no_grad():
                self.id_emb.weight[0].zero_()

        if mode in ("text", "hybrid", "randproj"):
            if mode == "randproj":
                # 용량 대조군: 텍스트 대신 무작위 초기화한 학습 테이블 (V+1, latent_dim).
                tm = torch.randn(num_items + 1, latent_dim) * 0.02
                self.text_matrix = nn.Parameter(tm)
                with torch.no_grad():
                    self.text_matrix[0].zero_()
            else:
                assert text_matrix is not None, "text 모드에는 text_matrix가 필요하다"
                tm = torch.as_tensor(text_matrix, dtype=torch.float32)  # (V+1, d_text)
                if train_text:
                    # 텍스트 임베딩 자체를 미세조정(end-to-end). 비싸지만 가능.
                    self.text_matrix = nn.Parameter(tm.clone())
                else:
                    # frozen: 의미 공간을 그대로 두고 projection만 학습한다.
                    self.register_buffer("text_matrix", tm)
            d_text = self.text_matrix.shape[1]
            self.text_proj = nn.Linear(d_text, d_model, bias=True)
            self.text_norm = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def _projected_text(self):
        """projection을 적용한 텍스트 행렬 (V+1, d_model), 행 0은 0으로 강제."""
        m = self.text_norm(self.text_proj(self.text_matrix))
        m = m.clone()
        m[0] = 0.0  # padding 행은 항상 0
        return m

    def matrix(self):
        """전체 아이템 표현 행렬 (V+1, d_model). 출력 scoring(weight tying)에 사용."""
        if self.mode == "id":
            return self.id_emb.weight
        if self.mode in ("text", "randproj"):
            return self._projected_text()
        # hybrid: 두 표현의 합
        m = self.id_emb.weight + self._projected_text()
        return m

    def forward(self, idx):
        """입력 인덱스 (B, L) -> 임베딩 (B, L, d_model). 입력 lookup에 사용."""
        if self.mode == "id":
            x = self.id_emb(idx)
        elif self.mode in ("text", "randproj"):
            # 선택된 행만 projection (전체 행렬 재계산 불필요)
            x = self.text_norm(self.text_proj(self.text_matrix[idx]))
            x = x * (idx != 0).unsqueeze(-1)  # pad 위치는 0
        else:  # hybrid
            xi = self.id_emb(idx)
            xt = self.text_norm(self.text_proj(self.text_matrix[idx]))
            xt = xt * (idx != 0).unsqueeze(-1)
            x = xi + xt
        return self.drop(x)


class SASRec(nn.Module):
    """단방향(causal) self-attention 시퀀스 추천 백본."""

    def __init__(self, item_repr, d_model, maxlen, n_layers=2, n_heads=2,
                 dropout=0.2):
        super().__init__()
        self.item_repr = item_repr
        self.d_model = d_model
        self.maxlen = maxlen
        self.pos_emb = nn.Embedding(maxlen, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.input_drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.last_norm = nn.LayerNorm(d_model)

    def seq_hidden(self, seq_idx):
        """입력 시퀀스 (B, L) -> 위치별 hidden state (B, L, d_model)."""
        B, L = seq_idx.shape
        positions = torch.arange(L, device=seq_idx.device).unsqueeze(0).expand(B, L)
        x = self.item_repr(seq_idx) + self.pos_emb(positions)
        x = self.input_drop(x)

        # causal mask: 미래 위치는 못 본다 (True = 차단)
        causal = torch.triu(
            torch.ones(L, L, device=seq_idx.device, dtype=torch.bool), diagonal=1)
        pad_mask = seq_idx == 0  # (B, L), True = padding key
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        return self.last_norm(h)

    def user_vector(self, seq_idx):
        """마지막 실제(non-pad) 위치의 표현 -> 사용자 벡터 (B, d_model).

        right-pad를 쓰므로 실제 토큰은 앞쪽에 모여 있다. 각 시퀀스의 마지막 실제
        위치 = (길이 - 1)을 골라낸다. 길이 0(빈 시퀀스)은 위치 0으로 clamp하지만,
        호출부에서 빈 history는 별도로 처리한다.
        """
        h = self.seq_hidden(seq_idx)
        lengths = (seq_idx != 0).sum(dim=1)            # (B,)
        last = (lengths - 1).clamp(min=0)
        return h[torch.arange(h.size(0), device=seq_idx.device), last]
