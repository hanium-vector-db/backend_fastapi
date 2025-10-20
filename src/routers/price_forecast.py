# src/api/routers/price_forecast.py  ← 경로 예시
from fastapi import APIRouter, Query, HTTPException
import os, pickle, math
import numpy as np
import pandas as pd
import torch

from src.modules.price_forecast.model import LSTMForecaster
from src.modules.price_forecast.data_utils import seasonal_resample, add_time_features
from src.modules.price_forecast.predict import predict_future  # 재귀 1-step을 내부서 쓰려면 predict_future 사용

router = APIRouter(tags=["Price Forecast"])  # ← 라우터 자체 prefix는 두지 않음(권장)

# 모델·스케일러 파일 경로
MODEL_DIR = "src/modules/price_forecast/models"
DATA_DIR  = "data"

LOOKBACK = 45
INTERP_MAX_GAP = 14

def _safe_read_csv(p):
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail=f"CSV not found: {p}")
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="cp949")

def _infer_horizon_from_state(state_dict: dict) -> int:
    w = state_dict.get("fc.weight")
    if w is None:
        raise HTTPException(status_code=500, detail="Invalid model state: fc.weight not found")
    return w.shape[0]

@router.get("/", summary="과일 가격 예측")
async def forecast_price(
    fruit: str = Query(..., description="과일 식별자 (예: Mango_1pc)"),
    horizon: int = Query(30, ge=1, le=90, description="예측 일수"),
    start_date: str = Query("auto", description="'YYYY-MM-DD' 또는 'auto'")
):
    # 경로 확인
    model_pt   = os.path.join(MODEL_DIR, f"{fruit}_best.model.pt")
    model_pth  = os.path.join(MODEL_DIR, f"{fruit}_Best.model.pth")
    scaler_pkl = os.path.join(MODEL_DIR, f"{fruit}_scaler.pkl")
    csv_path   = os.path.join(DATA_DIR,  f"{fruit}.csv")

    model_path = model_pt if os.path.exists(model_pt) else model_pth
    if not (model_path and os.path.exists(model_path) and os.path.exists(scaler_pkl)):
        raise HTTPException(status_code=404, detail=f"모델/스케일러 파일을 찾을 수 없습니다: {fruit}")

    # 데이터 로드 & 리샘플
    df = _safe_read_csv(csv_path)
    dcol, pcol = df.columns[0], df.columns[1]
    series = seasonal_resample(df, dcol, pcol, INTERP_MAX_GAP)
    if series.empty:
        raise HTTPException(status_code=400, detail="시계열 데이터가 비었습니다.")

    # 시작일 결정
    last_obs = series.index.max()
    start_dt = (last_obs + pd.Timedelta(days=1)) if start_date in ("auto", "", None) else pd.Timestamp(start_date)

    # 스케일러/모델 로드
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state  = torch.load(model_path, map_location=device)
    out_dim = _infer_horizon_from_state(state)
    model = LSTMForecaster(input_dim=7, hidden=128, layers=2, dropout=0.3, horizon=out_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    # 초기 lookback 윈도우 구성
    feats = add_time_features(series.index)
    frame = pd.concat([pd.DataFrame({'price': series}), feats], axis=1)
    frame['price_scaled'] = scaler.transform(frame[['price']].values)
    X_cols = ['price_scaled','sin_doy','cos_doy','sin_dow','cos_dow','month_norm','year_norm']
    desired_end = min(start_dt - pd.Timedelta(days=1), series.index.max())
    arr = frame[X_cols].loc[:desired_end].values
    valid = ~np.isnan(arr).any(axis=1)

    start_idx = None
    for pos in range(len(arr) - LOOKBACK, -1, -1):
        if valid[pos:pos+LOOKBACK].all():
            start_idx = pos
            break
    if start_idx is None:
        raise HTTPException(status_code=400, detail="LOOKBACK 윈도우 생성 실패(시작일 조정 또는 LOOKBACK 축소 필요)")

    # recursive rollout (1-step씩 horizon까지)
    preds = []
    win = arr[start_idx:start_idx+LOOKBACK].astype(np.float32)
    for step in range(horizon):
        X_in = torch.tensor(win).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(X_in).cpu().numpy().ravel()
        next_scaled = float(out[0])                  # 1-step ahead
        next_price  = scaler.inverse_transform(np.array([[next_scaled]])).ravel()[0]
        next_date   = start_dt + pd.Timedelta(days=step)

        # 다음 스텝 시간 피처 만들고 윈도우 갱신
        sin_doy = math.sin(2*math.pi*next_date.dayofyear/365.25)
        cos_doy = math.cos(2*math.pi*next_date.dayofyear/365.25)
        sin_dow = math.sin(2*math.pi*next_date.dayofweek/7.0)
        cos_dow = math.cos(2*math.pi*next_date.dayofweek/7.0)
        month_norm = next_date.month / 12.0
        yr_min, yr_max = series.index.year.min(), series.index.year.max()
        year_norm  = (next_date.year - yr_min) / max(1, (yr_max - yr_min))

        new_row = np.array([next_scaled, sin_doy, cos_doy, sin_dow, cos_dow, month_norm, year_norm], dtype=np.float32)
        win = np.vstack([win[1:], new_row])

        preds.append({"date": next_date.strftime("%Y-%m-%d"), "price": float(next_price)})

    return {
        "fruit": fruit,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "horizon": horizon,
        "predictions": preds
    }

