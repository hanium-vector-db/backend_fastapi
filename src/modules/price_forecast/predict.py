# predict.py
# -*- coding: utf-8 -*-
"""
미래 예측 모듈
"""
import numpy as np
import pandas as pd
import torch
from modules.data_utils import add_time_features


def predict_future(model, series, scaler, lookback, horizon_days, start_date_str, device):
    """
    Recursive 1-step 방식으로 미래 예측

    Args:
        model: 학습된 모델
        series: 과거 가격 시계열
        scaler: StandardScaler
        lookback: 입력 윈도우 길이
        horizon_days: 예측할 일수
        start_date_str: 시작 날짜 ('auto' 또는 'YYYY-MM-DD')
        device: torch.device

    Returns:
        pd.DataFrame: 예측 결과 (date, pred_price)
    """
    model.eval()

    # 시작일 결정
    last_obs = series.index[-1]
    if start_date_str in (None, "", "auto"):
        start = last_obs + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp(start_date_str)

    # 시간 특성 추가
    feats = add_time_features(series.index)
    frame = pd.concat([pd.DataFrame({'price': series}), feats], axis=1)
    frame['price_scaled'] = scaler.transform(frame[['price']].values)

    # 특성 컬럼
    X_cols = ['price_scaled', 'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow', 'month_norm', 'year_norm']

    # 시작일 이전까지의 데이터
    end_date = min(start - pd.Timedelta(days=1), last_obs)
    arr = frame[X_cols].loc[:end_date].values
    valid = ~np.isnan(arr).any(axis=1)

    # Lookback 윈도우 찾기 (뒤에서부터)
    start_idx = None
    for tt in range(len(arr) - lookback, -1, -1):
        if valid[tt:tt + lookback].all():
            start_idx = tt
            break

    if start_idx is None:
        raise ValueError("LOOKBACK 윈도우 생성 실패 (데이터 부족 또는 NaN 존재)")

    # 초기 윈도우
    window = arr[start_idx:start_idx + lookback].astype(np.float32)

    # Recursive 예측
    predictions = []
    dates = []

    for step in range(horizon_days):
        # 모델 입력
        X_in = torch.tensor(window).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(X_in).cpu().numpy().ravel()

        # 1-step 예측값 (스케일된 상태)
        next_scaled = float(out[0])

        # 실제 가격으로 변환
        next_price = scaler.inverse_transform(np.array([[next_scaled]])).ravel()[0]

        # 다음 날짜
        next_date = start + pd.Timedelta(days=step)

        # 다음 날짜의 시간 특성 계산
        sin_doy = np.sin(2 * np.pi * next_date.dayofyear / 365.25)
        cos_doy = np.cos(2 * np.pi * next_date.dayofyear / 365.25)
        sin_dow = np.sin(2 * np.pi * next_date.dayofweek / 7.0)
        cos_dow = np.cos(2 * np.pi * next_date.dayofweek / 7.0)
        month_norm = next_date.month / 12.0

        yr_min = series.index.year.min()
        yr_max = series.index.year.max()
        year_norm = (next_date.year - yr_min) / max(1, (yr_max - yr_min))

        # 새로운 행 생성
        new_row = np.array([
            next_scaled, sin_doy, cos_doy, sin_dow, cos_dow, month_norm, year_norm
        ], dtype=np.float32)

        # 윈도우 업데이트 (가장 오래된 값 제거, 새 값 추가)
        window = np.vstack([window[1:], new_row])

        # 결과 저장
        predictions.append(next_price)
        dates.append(next_date)

    return pd.DataFrame({'date': dates, 'pred_price': predictions})
