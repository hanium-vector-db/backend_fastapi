# dataset.py
# -*- coding: utf-8 -*-
"""
PyTorch Dataset 클래스
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from modules.data_utils import add_time_features


class SingleFruitDataset(Dataset):
    """
    단일 과일 시계열 데이터셋

    Args:
        series: 가격 시계열 (pd.Series)
        split_index: 사용할 인덱스 (train/val/test의 DatetimeIndex)
        lookback: 과거 윈도우 길이
        horizon: 예측 일수
        scaler: StandardScaler 객체 (fit된 상태)
    """

    def __init__(self, series, split_index, lookback, horizon, scaler):
        self.samples = []
        self.lookback = lookback
        self.horizon = horizon

        # 가격 데이터프레임 생성
        price = pd.DataFrame({'price': series})

        # 시간 특성 추가
        feats = add_time_features(price.index)

        # 병합
        frame = pd.concat([price, feats], axis=1)

        # 가격 스케일링
        frame['price_scaled'] = scaler.transform(frame[['price']].values)

        # 특성 컬럼 정의
        X_cols = ['price_scaled', 'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow', 'month_norm', 'year_norm']

        # 분할 인덱스에 해당하는 데이터만 선택
        chosen = frame.loc[frame.index.isin(split_index)]

        # numpy 배열로 변환
        arr = chosen[X_cols].values
        valid = ~np.isnan(arr).any(axis=1)

        # 윈도우 슬라이딩으로 샘플 생성
        for t in range(len(chosen) - lookback - horizon + 1):
            # 윈도우 전체가 유효한지 확인
            if valid[t:t + lookback + horizon].all():
                X = arr[t:t + lookback].astype(np.float32)
                y = chosen['price_scaled'].values[t + lookback:t + lookback + horizon].astype(np.float32)
                self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X), torch.tensor(y)
