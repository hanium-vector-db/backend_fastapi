# data_utils.py
# -*- coding: utf-8 -*-
"""
데이터 전처리 및 유틸리티 함수
"""
import numpy as np
import pandas as pd
from modules.config import LOOKBACK_MAX, HORIZON_MAX


def safe_read_csv(path):
    """
    CSV 파일을 안전하게 읽기 (인코딩 처리)
    """
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='cp949')


def seasonal_resample(df, date_col, price_col, max_gap):
    """
    일별 데이터로 리샘플링 및 보간

    Args:
        df: 원본 데이터프레임
        date_col: 날짜 컬럼명
        price_col: 가격 컬럼명
        max_gap: 최대 보간 공백 (일)

    Returns:
        pd.Series: 일별 리샘플링된 시계열
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df[price_col] = df[price_col].astype(str).str.replace(",", "", regex=False).astype(float)
    s = df.set_index(date_col)[price_col].resample('D').mean()
    # 짧은 공백만 보간 (제철 공백은 유지)
    return s.interpolate(method='time', limit=max_gap, limit_direction='both')


def add_time_features(index: pd.DatetimeIndex):
    """
    시간 관련 특성 추가 (계절성 및 트렌드)

    Returns:
        pd.DataFrame: 시간 특성 데이터프레임
    """
    tmp = pd.DataFrame(index=index)
    doy = index.dayofyear.values
    dow = index.dayofweek.values

    # 계절성: sin/cos 변환
    tmp['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    tmp['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    tmp['sin_dow'] = np.sin(2 * np.pi * dow / 7.0)
    tmp['cos_dow'] = np.cos(2 * np.pi * dow / 7.0)

    # 추가 특성
    tmp['month_norm'] = index.month / 12.0
    yr_min, yr_max = index.year.min(), index.year.max()
    tmp['year_norm'] = (index.year - yr_min) / max(1, (yr_max - yr_min))

    return tmp


def longest_valid_block(series: pd.Series):
    """
    NaN이 아닌 가장 긴 연속 구간 반환

    Args:
        series: 시계열 데이터

    Returns:
        pd.Series or None: 가장 긴 연속 블록
    """
    mask_arr = series.notna().to_numpy()
    if mask_arr.sum() == 0:
        return None

    idx = series.index
    blocks = []
    start_pos = None

    for pos in range(len(series)):
        if mask_arr[pos] and start_pos is None:
            start_pos = pos
        is_break = (not mask_arr[pos]) or (pos == len(series) - 1)
        if is_break and (start_pos is not None):
            end_pos = pos if mask_arr[pos] else pos - 1
            start_ts = idx[start_pos]
            end_ts = idx[end_pos]
            block = series.loc[start_ts:end_ts].dropna()
            if len(block) > 0:
                blocks.append(block)
            start_pos = None

    if not blocks:
        return None
    return max(blocks, key=lambda s: len(s))


def split_block_lenient(block: pd.Series, min_len: int, tried=("7:2:1", "8:1:1", "9:0.5:0.5")):
    """
    관대한 분할: 여러 비율로 시도하여 train/val/test 생성

    Args:
        block: 연속 블록
        min_len: 최소 길이
        tried: 시도할 비율 튜플

    Returns:
        tuple: (train, val, test) or (None, None, None)
    """
    def _split(blk: pd.Series, ratio: str):
        a, b, c = [float(x) for x in ratio.split(':')]
        tot = a + b + c
        a /= tot
        b /= tot
        idx = blk.index
        cut1 = int(len(idx) * a)
        cut2 = int(len(idx) * (a + b))
        tr = blk.iloc[:cut1]
        va = blk.iloc[cut1:cut2]
        te = blk.iloc[cut2:]
        return tr, va, te

    for r in tried:
        tr, va, te = _split(block, r)
        if len(tr) >= min_len and len(va) >= min_len and len(te) >= min_len:
            return tr, va, te
    return None, None, None


def adapt_window_for_block(block_len, base_lb=LOOKBACK_MAX, base_hz=HORIZON_MAX):
    """
    블록 길이에 따라 윈도우 크기 자동 조정

    Args:
        block_len: 블록 길이
        base_lb: 기본 lookback
        base_hz: 기본 horizon

    Returns:
        tuple: (lookback, horizon)
    """
    lb = min(base_lb, max(20, block_len // 4))  # 최소 20
    hz = min(base_hz, max(3, block_len // 20))  # 최소 3
    return lb, hz


def sMAPE(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error 계산
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)
