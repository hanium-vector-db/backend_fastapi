# model.py
# -*- coding: utf-8 -*-
"""
LSTM 기반 시계열 예측 모델
"""
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    LSTM 기반 다변량 시계열 예측 모델

    Args:
        input_dim: 입력 특성 개수 (기본 7: price_scaled + 시간 특성 6개)
        hidden: LSTM 은닉층 크기
        layers: LSTM 레이어 개수
        dropout: Dropout 비율
        horizon: 예측 일수
    """

    def __init__(self, input_dim=7, hidden=128, layers=2, dropout=0.3, horizon=7):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, lookback, input_dim)

        Returns:
            predictions: (batch_size, horizon)
        """
        # LSTM 출력: (batch_size, lookback, hidden)
        out, (h_n, c_n) = self.lstm(x)

        # 마지막 타임스텝의 출력 사용
        last_out = out[:, -1, :]  # (batch_size, hidden)

        # Fully Connected Layer
        predictions = self.fc(last_out)  # (batch_size, horizon)

        return predictions

    def get_num_params(self):
        """
        모델의 전체 파라미터 개수 반환
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
