# src/routers/price_forecast.py
from fastapi import APIRouter, Query, HTTPException
import os, pickle, torch
import pandas as pd
from datetime import datetime
from src.modules.price_forecast.model import LSTMForecaster
from src.modules.price_forecast.data_utils import seasonal_resample, add_time_features
from src.modules.price_forecast.predict import predict_future

router = APIRouter(prefix="/predict-price", tags=["Price Forecast"])

# 모델·스케일러 파일 경로 설정
MODEL_DIR = "src/modules/price_forecast/models"

@router.get("/")
async def forecast_price(
    fruit: str = Query(..., description="과일 이름 (예: Mango_1pc)"),
    horizon: int = Query(30, description="예측 일수 (기본 30일)"),
    start_date: str = Query("2025-08-01", description="예측 시작일 (auto 가능)")
):
    """
    학습된 모델과 스케일러를 로드하여 미래 가격을 예측하는 API
    """
    model_path = os.path.join(MODEL_DIR, f"{fruit}_best.model.pt")
    scaler_path = os.path.join(MODEL_DIR, f"{fruit}_scaler.pkl")
    csv_path = f"data/{fruit}.csv"  # 실제 데이터 경로 (또는 DB)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail=f"모델 또는 스케일러가 없습니다: {fruit}")

    df = pd.read_csv(csv_path)
    date_col, price_col = df.columns[0], df.columns[1]
    series = seasonal_resample(df, date_col, price_col, 14)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = LSTMForecaster(input_dim=7, hidden=128, layers=2, dropout=0.3, horizon=7)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    try:
        future_df = predict_future(model, series, scaler, lookback=45,
                                   horizon_days=horizon, start_date_str=start_date, device="cpu")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

    return {
        "fruit": fruit,
        "start_date": start_date,
        "horizon": horizon,
        "predictions": future_df.to_dict(orient="records")
    }
